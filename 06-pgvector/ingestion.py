# Embeddings and then Ingesting
# Ingesting pipeline: You'll hear it a lot in RAG contexts specifically, "ingestion pipeline" is the standard term for "chunk documents → embed → store,
# Ingestion is periodic/batch that must happen in advance (offline), Retrieval pipeline is real-time, per-request (user prompt)
# ingestion pipelines need idempotency — running the same batch twice shouldn't double your data.
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
import hashlib

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

model = SentenceTransformer("all-MiniLM-L6-v2")

pgConnection = psycopg.connect(
    "postgresql://postgres:localdev@localhost:5432/agentic_learning"
)

# psycopg: Teaches how to send/receive the vector type
# it lets you pass a raw numpy array (what model.encode() returns) straight into a SQL parameter instead of manually converting it to a Postgres array literal string. it's the actual bridge between "Python embedding" and "Postgres vector column."
register_vector(pgConnection)

# Ofc the dataset here can be anything from customer Confluence contents to their help side docs, etc
# same kind of test sentences as 05, now tagged with a some metadata fields e.g tenant
docs = [
    # nordic-mfg: same underlying fault, two languages
    {"content": "Hydraulic pump pressure drop on line 3", "tenant_id": "nordic-mfg", "source": "field-report-2024-03", "language": "en"},
    {"content": "Tryckfall i hydraulpumpen på linje 3", "tenant_id": "nordic-mfg", "source": "field-report-2024-03-sv", "language": "sv"},
    # nordic-mfg: different fault, same machine family
    {"content": "Conveyor belt motor overheating after 6 hours runtime", "tenant_id": "nordic-mfg", "source": "field-report-2024-05", "language": "en"},
    # nordic-mfg: unrelated to hydraulics/conveyor, tests topic separation
    {"content": "Annual safety inspection checklist for warehouse forklifts", "tenant_id": "nordic-mfg", "source": "checklist-2024", "language": "en"},
    # eu-insurer: claims triage domain
    {"content": "Water damage claim, kitchen pipe burst", "tenant_id": "eu-insurer", "source": "claim-88213", "language": "en"},
    {"content": "Rear-end collision, minor bumper damage, no injuries reported", "tenant_id": "eu-insurer", "source": "claim-88240", "language": "en"},
    # eu-insurer: same claim type, different language (Swedish market)
    {"content": "Vattenskada, rör som sprack i köket", "tenant_id": "eu-insurer", "source": "claim-88213-sv", "language": "sv"},
]

with pgConnection.cursor() as cur:
    for doc in docs:
        h = content_hash(doc["content"])

        cur.execute(
            "SELECT id FROM documents WHERE tenant_id = %s AND content_hash = %s",
            (doc["tenant_id"], h),
        )
        if cur.fetchone():
            print(f"Skipping unchanged: {doc['source']}")
            continue  # no re-embedding, no write

        embedding = model.encode(doc["content"])
        # the text you want semantic search over. Metadata (tenant_id, source, language, timestamps) is stored as plain fields alongside the vector, never passed through the embedding model.
        cur.execute(
            """
            INSERT INTO documents (content, embedding, tenant_id, source, language, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (tenant_id, content_hash) DO NOTHING
            """,
            (doc["content"], embedding, doc["tenant_id"], doc["source"], doc["language"], h),
        )
pgConnection.commit()
pgConnection.close()
print(f"Inserted {len(docs)} documents")

# ..


# To confirm the ingesting and dimension of the results, run:
# SELECT id, vector_dims(embedding) FROM documents;
# vector_dims() is a pgvector-provided function — every row should show 384. If any row shows something else or NULL, that row's embedding didn't get written correctly.