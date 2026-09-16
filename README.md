# Agentic AI: EM to FDE journey

Hands-on learning log, building up from a single API call to full agent loops and retrieval.

- `01-first-call`: one API call, no framework
- `02-structured-output`: typed output with Pydantic
- `03-tool-calling`: the model requests a function, and the code decide wether to run it or not
- `04-agent-loop`: multi-step loop with a step cap
- `05-embeddings-vectors`: semantic search from scratch, no LLM involved

## Get started

This is a Python with OpenAI LLM stack.

Put `OPENAI_API_KEY` in `.env`.

```bash
# Exercise 1
uv run 01-first-call/main.py
# Exercise 2
uv run 02-structured-output/
# ...
```
