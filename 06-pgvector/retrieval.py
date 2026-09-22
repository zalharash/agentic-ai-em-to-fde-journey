# Purpose of this script:
# It's the retrieval half of RAG, isolated and testable on its own, before you wire it into an agent.
# The core mechanism is Cosine similarity/distance, ordered smallest-distance-first, top-k, is genuinely the foundational retrieval primitive in essentially every production RAG system.

# The problem: what's missing from that list:
# The Swedish translation of your #1 result — Tryckfall i hydraulpumpen på linje 3, describing the exact same fault — isn't in the top 3 at all. It scored worse than an unrelated insurance claim in a different domain entirely. Same meaning, wrong language, and the model treated it as less relevant than something with zero topical overlap.
# The fix:
# Conceptually (not yet done): swap all-MiniLM-L6-v2 for a model explicitly trained for cross-lingual matching — e.g. paraphrase-multilingual-MiniLM-L12-v2 — and re-run the same query to see whether the Swedish sentence now ranks near its English twin.
# for a real multilingual deployment, you'd need a multilingual-specific embedding model — e.g., paraphrase-multilingual-MiniLM-L12-v2 or similar — not this one.

# Critical detail: Both it must be the same model as ingestion — if query and stored docs were embedded by different models, the vectors would live in different, incomparable spaces, and distance would be meaningless.

# --- PRODUCTION NOTE: Row-Level Security (RLS) not yet enabled ---
# Current isolation relies entirely on the app remembering to add WHERE tenant_id = %s
# to every query. This is a real risk: one missed filter = cross-tenant data leak.
#
# In production, RLS should be enabled at the table level so Postgres enforces
# isolation even if application code forgets the filter:
#
#   ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
#
#   CREATE POLICY tenant_isolation ON documents
#       USING (tenant_id = current_setting('app.current_tenant'));
#
# Then, per-connection/session, the app sets which tenant it's acting as:
#
# SET app.current_tenant = 'nordic-mfg';
#
# After that, ANY query against `documents` — even a bare `SELECT * FROM documents`
# with no WHERE clause at all — will only ever see nordic-mfg rows. The filter
# becomes structurally impossible to bypass, not just a convention to remember.
# -------------------------------------------------------------------
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_connection():
    conn = psycopg.connect(
        "postgresql://postgres:localdev@localhost:5432/agentic_learning"
    )
    register_vector(conn)
    return conn


def print_results(rows):
    for content, tenant_id, source, language, distance in rows:
        print(f"[{distance:.4f}] ({tenant_id}/{language}) {source}: {content}")


def search_unfiltered(conn, query_embedding, limit=3):
    """Original baseline query — no metadata filter, full corpus."""
    # For every row in documents, compute the cosine distance between that row's stored embedding and the query vector you just passed in. embedding <=> %s AS distance
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, tenant_id, source, language,
                   embedding <=> %s AS distance
            FROM documents
            ORDER BY distance
            LIMIT %s
            """,
            (query_embedding, limit),
        )
        return cur.fetchall()

# filtering doesn't improve the embedding's judgment, it structurally removes ineligible candidates before ranking runs.
def search_by_tenant(conn, query_embedding, tenant_id, limit=3):
    """Filter to a single tenant before ranking — tests isolation."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, tenant_id, source, language,
                   embedding <=> %s AS distance
            FROM documents
            WHERE tenant_id = %s
            ORDER BY distance
            LIMIT %s
            """,
            (query_embedding, tenant_id, limit),
        )
        return cur.fetchall()


def search_by_tenant_and_language(conn, query_embedding, tenant_id, language, limit=3):
    """Filter to tenant + language, then rank."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, tenant_id, source, language,
                   embedding <=> %s AS distance
            FROM documents
            WHERE tenant_id = %s AND language = %s
            ORDER BY distance
            LIMIT %s
            """,
            (query_embedding, tenant_id, language, limit),
        )
        return cur.fetchall()


def main():
    conn = get_connection()
    query_text = "pump losing pressure and overheating"
    query_embedding = model.encode(query_text)

    print("=== Unfiltered (baseline) ===")
    print_results(search_unfiltered(conn, query_embedding))

    print("\n=== Filtered: tenant_id = nordic-mfg ===")
    # filtering removes ineligible candidates (that are not in nordic-mfg tenant) before ranking runs.
    print_results(search_by_tenant(conn, query_embedding, "nordic-mfg"))

    print("\n=== Filtered: tenant_id = nordic-mfg, language = en ===")
    # Tenant + language filtered: correctly excludes the Swedish document entirely
    print_results(search_by_tenant_and_language(conn, query_embedding, "nordic-mfg", "en"))

    conn.close()


if __name__ == "__main__":
    main()


# Your output, decoded:
# query_text = "pump losing pressure and overheating"
# === Unfiltered (baseline) ===
# [0.4731] (nordic-mfg/en) field-report-2024-03: Hydraulic pump pressure drop on line 3
# [0.4991] (nordic-mfg/en) field-report-2024-05: Conveyor belt motor overheating after 6 hours runtime
# [0.8149] (eu-insurer/en) claim-88213: Water damage claim, kitchen pipe burst

# === Filtered: tenant_id = nordic-mfg ===
# [0.4731] (nordic-mfg/en) field-report-2024-03: Hydraulic pump pressure drop on line 3
# [0.4991] (nordic-mfg/en) field-report-2024-05: Conveyor belt motor overheating after 6 hours runtime
# [0.8918] (nordic-mfg/sv) field-report-2024-03-sv: Tryckfall i hydraulpumpen på linje 3

# === Filtered: tenant_id = nordic-mfg, language = en ===
# [0.4731] (nordic-mfg/en) field-report-2024-03: Hydraulic pump pressure drop on line 3
# [0.4991] (nordic-mfg/en) field-report-2024-05: Conveyor belt motor overheating after 6 hours runtime
# [1.0416] (nordic-mfg/en) checklist-2024: Annual safety inspection checklist for warehouse forklifts
