# 07-pinecone/query.py
#
# Mirrors 06-pgvector/query.py structure, swapping Postgres+Docker for Pinecone's hosted API.
#
# SETUP NEEDED BEFORE RUNNING (not yet done — reference only):
# 1. Sign up at https://www.pinecone.io (free tier available — no local install, no Docker)
# 2. Create an API key from the Pinecone console
# 3. Create an index via the console or code — must set dimension=384 to match MiniLM's output
#    (Postgres let you define the column type yourself; Pinecone makes you declare
#    the vector dimension up front when creating the index — same constraint, different place)
# 4. pip/uv add the SDK:  uv add pinecone
# 5. Add PINECONE_API_KEY to .env

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_NAME = "agentic-learning"


def get_index():
    """
    Pinecone equivalent of get_connection() + CREATE TABLE.
    No Docker container to run — this connects to Pinecone's hosted service directly.
    create_index() only needs to run once ever, not per script run (like CREATE TABLE).
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # must match MiniLM's output size, same constraint as vector(384) in pgvector
            metric="cosine",  # equivalent to choosing the <=> operator in pgvector
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # hosted infra choice — no equivalent in local Docker setup
        )

    return pc.Index(INDEX_NAME)


def ingest_docs(index, docs):
    """
    Pinecone equivalent of ingest.py's INSERT loop.
    upsert() is Pinecone's native idempotency mechanism: same id = overwrite, no duplicates.
    Unlike pgvector, we didn't need to build our own content-hash uniqueness constraint —
    Pinecone's upsert-by-id is idempotent by design, as long as you assign stable ids yourself.
    """
    vectors = []
    for doc in docs:
        embedding = model.encode(doc["content"])
        vectors.append({
            "id": doc["source"],  # Pinecone requires a stable id per vector — we reuse "source" as the natural key
            "values": embedding.tolist(),  # Pinecone wants a plain list, not a numpy array
            "metadata": {
                "content": doc["content"],  # Pinecone has no separate "content" column — raw text lives inside metadata
                "tenant_id": doc["tenant_id"],
                "language": doc["language"],
            },
        })
    index.upsert(vectors=vectors)


def print_results(matches):
    for match in matches:
        md = match["metadata"]
        # Pinecone returns a similarity SCORE (higher = more similar) when metric="cosine",
        # the inverse framing of pgvector's DISTANCE (lower = more similar) — easy mix-up, worth noting
        print(f"[{match['score']:.4f}] ({md['tenant_id']}/{md['language']}) {match['id']}: {md['content']}")


def search_unfiltered(index, query_embedding, top_k=3):
    """Baseline query — no metadata filter, full index."""
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
    )
    return results["matches"]


def search_by_tenant(index, query_embedding, tenant_id, top_k=3):
    """
    Filtered query — Pinecone's filter dict is its DQL: JSON-style operators
    ($eq, $in, $gte, etc.) instead of SQL's WHERE clause.
    """
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        filter={"tenant_id": {"$eq": tenant_id}},
        include_metadata=True,
    )
    return results["matches"]


def search_by_tenant_and_language(index, query_embedding, tenant_id, language, top_k=3):
    """Compound filter — combine two conditions with $and, Pinecone's equivalent of SQL's AND."""
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        filter={
            "$and": [
                {"tenant_id": {"$eq": tenant_id}},
                {"language": {"$eq": language}},
            ]
        },
        include_metadata=True,
    )
    return results["matches"]


def main():
    index = get_index()

    # same 7 docs as 06-pgvector, reused here for apples-to-apples comparison
    docs = [
        {"content": "Hydraulic pump pressure drop on line 3", "tenant_id": "nordic-mfg", "source": "field-report-2024-03", "language": "en"},
        {"content": "Tryckfall i hydraulpumpen på linje 3", "tenant_id": "nordic-mfg", "source": "field-report-2024-03-sv", "language": "sv"},
        {"content": "Conveyor belt motor overheating after 6 hours runtime", "tenant_id": "nordic-mfg", "source": "field-report-2024-05", "language": "en"},
        {"content": "Annual safety inspection checklist for warehouse forklifts", "tenant_id": "nordic-mfg", "source": "checklist-2024", "language": "en"},
        {"content": "Water damage claim, kitchen pipe burst", "tenant_id": "eu-insurer", "source": "claim-88213", "language": "en"},
        {"content": "Rear-end collision, minor bumper damage, no injuries reported", "tenant_id": "eu-insurer", "source": "claim-88240", "language": "en"},
        {"content": "Vattenskada, rör som sprack i köket", "tenant_id": "eu-insurer", "source": "claim-88213-sv", "language": "sv"},
    ]
    ingest_docs(index, docs)

    query_text = "pump losing pressure and overheating"
    query_embedding = model.encode(query_text)

    print("=== Unfiltered (baseline) ===")
    print_results(search_unfiltered(index, query_embedding))

    print("\n=== Filtered: tenant_id = nordic-mfg ===")
    print_results(search_by_tenant(index, query_embedding, "nordic-mfg"))

    print("\n=== Filtered: tenant_id = nordic-mfg, language = en ===")
    print_results(search_by_tenant_and_language(index, query_embedding, "nordic-mfg", "en"))


if __name__ == "__main__":
    main()