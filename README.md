# Agent demo

A house hunting agent: finds listings, rates fit against a buyer profile,
and drafts (never sends) a tailored viewing request.

- **LangGraph** ReAct loop with an explicit **self-correction** step: every
  tool result passes through a critic node that detects tool errors and
  injects a corrective instruction (bounded retries) instead of looping
  forever on the same broken call.
- **Claude Opus 4.8** as the primary model, falling back to **Claude Haiku
  4.5** automatically via LangChain's `.with_fallbacks(...)`.
- **LLM gateway** (LiteLLM proxy) in front of both models -- the agent holds
  only a gateway key; Anthropic credentials, routing, and per-model config
  live in `litellm_config.yaml` on the gateway, not in the app.
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
       -> agent_demo/llm.py                  # Opus 4.8 -> Haiku 4.5 fallback, via LLM gateway
            -> litellm proxy (:4000)         # holds ANTHROPIC_API_KEY, routes to Anthropic
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

Requires Python 3.13, `uv`, and Docker (for local Mongo/Postgres/the LLM
gateway).

```sh
cp .env.example .env        # fill in ANTHROPIC_API_KEY and LLM_GATEWAY_API_KEY at minimum
docker compose up -d        # local MongoDB + Postgres + LiteLLM gateway
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

`max_react_steps` and `max_self_correction_retries` (process-wide defaults in
`agent_demo/config.py`) can also be overridden per-request by adding either
field, e.g. `"max_react_steps": 4`, to the JSON body above -- useful for a
"quick look" call that should stop sooner than the default.

Web search defaults to the reference Brave Search MCP server -- set
`BRAVE_API_KEY` in `.env`, or swap `MCP_WEB_SEARCH_COMMAND`/`MCP_WEB_SEARCH_ARGS`
for a different MCP-compatible provider (Exa, Tavily, ...). Tracing
(`ARIZE_SPACE_ID`/`ARIZE_API_KEY`) and the AgentCore deployment path are both
optional for local dev -- everything else runs against real Anthropic/Mongo/
Postgres/MCP services with just the steps above.

### LLM gateway

`agent_demo/llm.py` never calls Anthropic directly -- it talks to a LiteLLM
proxy (`LLM_GATEWAY_BASE_URL`, default `http://localhost:4000`) over the
OpenAI-compatible API via `ChatOpenAI`, and authenticates with
`LLM_GATEWAY_API_KEY`. The `docker compose up -d` step above starts that
proxy from `litellm_config.yaml`, which is where `ANTHROPIC_API_KEY` and the
`PRIMARY_MODEL`/`FALLBACK_MODEL` -> real-model mapping live -- the app
process never sees the Anthropic key. To add a model or switch providers,
edit `litellm_config.yaml`'s `model_list` (and the matching env var), not
`agent_demo/llm.py`. In production, point `LLM_GATEWAY_BASE_URL` at a
centrally-hosted LiteLLM (or other OpenAI-compatible) gateway instead of the
local container.

## Testing

The graph's control flow (self-correction, retry give-up, step-budget
grace turn) is tested in isolation against a fake model and LangGraph's
in-memory checkpointer/store -- no Mongo/Postgres/MCP/Anthropic services
required:

```sh
uv sync --group dev
uv run pytest
```

## Known limitations

- **Prompt injection surface.** `buyer_profile` and the user's `message`
  (see `InvokeRequest` in `agent_demo/runner.py`) are untrusted text that
  flows directly into the system/human prompt of an agent with real side
  effects: it writes to Postgres (`house_listings`, `listing_ratings`,
  `viewing_requests`) and to long-term/factual memory in Mongo. The same
  applies to content pulled in by the `fetch` MCP tool -- a scraped listing
  page is also untrusted text the model reads. Nothing here defends against
  a crafted profile or page trying to steer the agent into unwanted tool
  calls (e.g. spamming `draft_viewing_request`). The design keeps the worst
  case bounded -- viewing requests are always drafts a human must send, and
  `max_react_steps`/`max_self_correction_retries` cap runaway tool use --
  but there's no explicit injection defense (e.g. tagging/quoting untrusted
  content, or a guardrail pass) beyond that.

## Deploying

See `deployment/README.md`.
