# Agent demo

A house hunting agent: finds listings, rates fit against a buyer profile,
and drafts (never sends) a tailored viewing request.

- **LangGraph** ReAct loop with an explicit **self-correction** step: every
  tool result passes through a critic node that detects tool errors and
  injects a corrective instruction (bounded retries) instead of looping
  forever on the same broken call.
- **Claude Opus 4.8** as the primary model, falling back to **Claude Haiku
  4.5** automatically via LangChain's `.with_fallbacks(...)`.
- **MCP servers** for web search and page fetching (swappable provider).
- **MongoDB** for three distinct memory tiers: short-term (per-session
  conversation state, via a LangGraph checkpointer), long-term (durable
  cross-session buyer preferences, via a custom `BaseStore`), and
  factual (a free-text research notes/knowledge base).
- **Postgres** for structured, queryable state: discovered house listings,
  fit ratings, and viewing-request drafts.
- **Arize AX** tracing via OpenInference/OpenTelemetry auto-instrumentation.
- Hosted on **AWS Bedrock AgentCore Runtime** (`main.py`); the core logic in
  `agent_demo/runner.py` is plain async Python, fully testable without it.

Requesting a viewing means the agent drafts a tailored inquiry message and
buyer highlights and saves them to Postgres for the buyer to review -- it
never contacts the listing agent or schedules anything itself. See
`deployment/README.md` for why, and for the full AgentCore deployment path.

## Architecture

```
main.py (AgentCore entrypoint)
  -> agent_demo/runner.py: run(payload)      # validate, build stack, invoke graph
       -> agent_demo/llm.py                  # Opus 4.8 -> Haiku 4.5 fallback
       -> agent_demo/graph/graph.py          # StateGraph: agent <-> tools -> critic
       -> agent_demo/tools/
            mcp_tools.py                     # web_search, fetch (via MCP)
            postgres_tools.py                # save/rate listings, draft viewing requests
            memory_tools.py                  # remember/recall facts + preferences
       -> agent_demo/memory/
            short_term.py                    # MongoDBSaver checkpointer (per session)
            long_term.py                     # MongoLongTermStore (per buyer)
            factual.py                       # FactualMemory (research notes)
       -> agent_demo/tracing/setup.py        # Arize AX / OpenInference
```

## Local development

Requires Python 3.13, `uv`, and Docker (for local Mongo/Postgres).

```sh
cp .env.example .env        # fill in ANTHROPIC_API_KEY at minimum
docker compose up -d        # local MongoDB + Postgres
uv sync

# Invoke the agent directly, no HTTP layer:
uv run python scripts/run_local.py "Find me a 3-bedroom house in Lisbon"

# Or serve the same /invocations + /ping contract AgentCore Runtime uses:
uv run python -m main
curl -X POST localhost:8080/invocations -H 'content-type: application/json' -d '{
  "message": "Find me a 3-bedroom house in Lisbon",
  "buyer_id": "buyer_123",
  "buyer_profile": "Family of three, budget EUR 450k, wants a garden and good transit access"
}'
```

Web search defaults to the reference Brave Search MCP server -- set
`BRAVE_API_KEY` in `.env`, or swap `MCP_WEB_SEARCH_COMMAND`/`MCP_WEB_SEARCH_ARGS`
for a different MCP-compatible provider (Exa, Tavily, ...). Tracing
(`ARIZE_SPACE_ID`/`ARIZE_API_KEY`) and the AgentCore deployment path are both
optional for local dev -- everything else runs against real Anthropic/Mongo/
Postgres/MCP services with just the steps above.

## Deploying

See `deployment/README.md`.
