# 09 — MCP Server: exposing search_documents as a standard, client-agnostic tool

## What we're doing

Stages 01–04 built a hand-rolled agent loop where tools were plain Python
functions, only callable by that one script. Stage 06 built a working
retrieval function (`search_by_tenant`) against pgvector, but it was still
just a function you called from Python directly.

This stage wraps that same retrieval logic as an **MCP server** — a small
process that exposes `search_documents` using the Model Context Protocol
(MCP), a standardized way for *any* MCP-compatible AI client to discover
the tool, read its description, and call it — without that client needing
any custom integration code for your specific setup.

This is the same underlying idea as `04-agent-loop`'s tool-calling pattern,
just standardized: instead of "my one agent loop knows about my one tool,"
it becomes "any MCP client can find and use this tool."

## Expected outcome

By the end of this stage:
- A local server (`server.py`) runs and exposes one tool: `search_documents`
- MCP Inspector (a browser-based test client) can call it directly and get
  back real results from your pgvector database
- Claude Desktop, a real production AI client, can discover the tool,
  decide on its own when to call it based on a natural-language prompt,
  and synthesize an answer from the raw results

## How the AI client actually uses it

1. On startup, the client reads your MCP config and starts `server.py` as
   a subprocess (this is the **stdio** transport — client and server talk
   over stdin/stdout, same machine, no network involved)
2. The client sends `tools/list` — your server responds with `search_documents`'s
   name, parameters, and description, auto-generated from your Python
   type hints and docstring
3. When you write a prompt, the model reads that tool description (as part
   of its context) and **decides for itself** whether calling the tool
   would help answer your question — you never invoke it manually
4. If it decides to call the tool, the client sends a `tools/call` request
   with the arguments the model chose; your server runs the actual
   Postgres query and returns the result as text
5. That result gets fed back into the model's context, and it writes the
   final answer — this is the same tool-result-back-into-context loop from
   `04-agent-loop`, just running through MCP's protocol instead of your
   own while-loop

## Setup — run it locally

```bash
mkdir 09-mcp-server
cd 09-mcp-server
```

Create `server.py` (wraps `06-pgvector`'s tenant-filtered search as an
`@mcp.tool()`-decorated function — see repo for full code).

Install dependencies — **pin to v1**, since v2 (released mid-2026) renamed
`FastMCP` to `MCPServer` and reshaped several APIs; v1 stays maintained
with security fixes and is simpler for learning the core concept:

```bash
uv add "mcp<2"
```

Postgres must be running (from `06-pgvector`) whenever this server is used:

```bash
docker start pgvector-learning
```

## Test it locally in the browser (MCP Inspector)

MCP Inspector is a browser-based test client — good for confirming the
server works *before* wiring it into a real AI client.

```bash
uv run mcp dev server.py
```

This opens a local web UI (`localhost`, port varies). Steps:

1. Click **Connect** — should show a green "Connected" status and list
   your server (`field-service-search`)
2. Click the **Tools** tab (not Prompts — this server defines no prompts)
3. Select `search_documents`, fill in the auto-generated form:
   - `query`: `pump losing pressure and overheating`
   - `tenant_id`: `nordic-mfg`
   - `limit`: `3`
4. Run it — confirm the returned distances/ranking match what
   `06-pgvector/query.py` returned directly. If they match exactly, the
   MCP wrapper is a faithful pass-through with nothing lost in translation.

**Gotcha:** if you get `Connection refused` on port 5432, Postgres isn't
running — `docker start pgvector-learning` and retry.

## Attach it to Claude Desktop

Claude Desktop has first-class, native MCP support — built by the team
that created the protocol, so this path is the most reliable and the one
worth learning first.

1. Edit the config file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Add your server:

```json
{
  "mcpServers": {
    "field-service-search": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/09-mcp-server", "run", "server.py"]
    }
  }
}
```

3. Fully quit and restart Claude Desktop (not just close the window)
4. Start a new conversation, check the tools/connector icon —
   `field-service-search` should be listed
5. Test with a natural prompt: *"Search nordic-mfg documents for a pump
   losing pressure and overheating"* — watch whether Claude decides to
   call the tool on its own, and how it phrases the synthesized answer

## Attach it to ChatGPT (note before attempting)

As of testing this stage (2026), ChatGPT's MCP support is real but
meaningfully different from Claude Desktop's:
- It sits behind **Developer Mode**, a setting most users don't have
  enabled by default, and is gated to certain plans
- Its primary path favors **hosted/remote MCP servers** (a public URL
  OpenAI calls on your behalf) over local stdio servers — local stdio
  servers are supported but are the less-traveled path and may need a
  compatibility shim depending on protocol version
- Treat this as a "worth trying later, budget extra troubleshooting time"
  path rather than a same-effort alternative to Claude Desktop

## Notes / gotchas

- pgvector must be running (`docker start pgvector-learning`) before any
  client — inspector or Claude Desktop — can successfully call the tool
- `mcp<2` pinned deliberately: v2 (MCPServer rename, reshaped Context
  injection, stricter validation) is a real API, but a moving target not
  worth chasing for this learning stage
- The MCP wrapper adds zero logic of its own — it's a pure pass-through
  to the same SQL query built in `06-pgvector`; any future accuracy
  improvements (better embedding model, reranking) happen there, not here