# 09-mcp-server/server.py
#
# Wraps the search_by_tenant logic from 06-pgvector as an MCP tool,
# callable by any MCP-compatible client (Claude Desktop, Claude Code, etc.)

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

model = SentenceTransformer("all-MiniLM-L6-v2")
mcp = FastMCP("field-service-search")  # name shown to MCP clients


def get_connection():
    conn = psycopg.connect(
        "postgresql://postgres:localdev@localhost:5432/agentic_learning"
    )
    register_vector(conn)
    return conn


@mcp.tool()
def search_documents(query: str, tenant_id: str, limit: int = 3) -> str:
    """
    Search field service / claims documents by semantic similarity,
    scoped to a single tenant.

    Args:
        query: natural language search query
        tenant_id: which tenant's documents to search (e.g. "nordic-mfg", "eu-insurer")
        limit: max number of results to return
    """
    query_embedding = model.encode(query)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, source, language,
                       embedding <=> %s AS distance
                FROM documents
                WHERE tenant_id = %s
                ORDER BY distance
                LIMIT %s
                """,
                (query_embedding, tenant_id, limit),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return "No matching documents found."

    return "\n".join(
        f"[{dist:.4f}] {source} ({lang}): {content}"
        for content, source, lang, dist in rows
    )


if __name__ == "__main__":
    mcp.run()  # starts the stdio server, waits for a client to connect