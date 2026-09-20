# 06 — pgvector: persistent embeddings + metadata filtering

Goal: move from in-memory embeddings (05) to persistent storage in Postgres,
add metadata filtering, and touch Row-Level Security for tenant isolation.

## Setup

1. Run Postgres with pgvector pre-built:

```bash
docker run --name pgvector-learning \
  -e POSTGRES_PASSWORD=localdev \
  -e POSTGRES_DB=agentic_learning \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

2. Confirm it's running:

```bash
docker ps
```
Start the container before running it with exec
```bash
docker start pgvector-learning
```

3. Enable the extension (once per database):

```bash
docker exec -it pgvector-learning psql -U postgres -d agentic_learning
```

Then inside psql:

```sql
CREATE EXTENSION vector;
```

`\dt` to confirm, `\q` to exit.

4. Python deps (note: quote psycopg[binary] in zsh or it globs):

```bash
uv add "psycopg[binary]" pgvector
```

## Table structure

```SQL
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    tenant_id TEXT NOT NULL,
    source TEXT,
    language TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

## Container lifecycle

- Stop: `docker stop pgvector-learning`
- Start again: `docker start pgvector-learning`
- Full reset (wipes data): `docker rm -f pgvector-learning` thene-run step 1

## Notes / gotchas
- (fill in as we hit them)
