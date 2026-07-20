# Agent demo

A platform harness that hosts multiple LangGraph agents behind one deployment
(`main.py`, selected via `AGENT_ID`), plus two agents built on it:

- **house-search** (the original agent): finds house listings, rates fit
  against a buyer profile, and drafts (never sends) a tailored viewing
  request.
- **faq-agent**: a deliberately minimal second agent (one static-answer tool,
  no Postgres/MCP) that exists to prove a new agent can be onboarded without
  touching `agent_demo/platform/` or `main.py`.

The point of the split: cross-cutting operational behavior -- the kill
switch, Arize AX tracing, LLM gateway routing, and step/retry budget
ceilings -- is owned entirely by `agent_demo/platform/` and enforced by
`agent_demo/platform/harness.py`, which is the only thing that ever calls an
agent's code. An agent supplies its own request schema, state, system
prompt, and tools via `agent_demo.platform.spec.AgentSpec`; it has no way to
skip the kill-switch check, bypass tracing, or call a model provider
directly, because it never runs its own entrypoint.

- **LangGraph** ReAct loop with an explicit **self-correction** step,
  provided by the platform's `graph_factory.build_react_graph`: every tool
  result passes through a critic node that detects tool errors and injects a
  corrective instruction (bounded retries) instead of looping forever on the
  same broken call.
- **Claude Opus 4.8** as each agent's primary model (its own choice, in its
  own config), falling back to **Claude Haiku 4.5** automatically via
  LangChain's `.with_fallbacks(...)`.
- **LLM gateway** (LiteLLM proxy) in front of both models -- an agent only
  ever picks model *names*; `agent_demo/platform/llm.py` is the only place a
  model client is constructed, and it always goes through the gateway.
  Anthropic credentials, routing, and per-model config live in
  `litellm_config.yaml` on the gateway, not in the app.
- **MCP servers** for web search and page fetching (swappable provider;
  house-search's choice, not the platform's).
- **MongoDB** for three distinct memory tiers, shared infrastructure any
  agent may use (`agent_demo/lib/memory/`): short-term (per-session
  conversation state, via a LangGraph checkpointer), long-term (durable
  cross-session preferences, via a custom `BaseStore`), and factual (a
  free-text research notes/knowledge base).
- **Postgres** for house-search's structured, queryable state: discovered
  listings, fit ratings, and viewing-request drafts.
- **Arize AX** tracing via OpenInference/OpenTelemetry auto-instrumentation,
  configured once by the platform for whichever agent is running.
- Hosted on **AWS Bedrock AgentCore Runtime** (`main.py`); the core logic in
  `agent_demo/platform/harness.py` is plain async Python, fully testable
  without it.

Requesting a viewing means house-search drafts a tailored inquiry message
and buyer highlights and saves them to Postgres for the buyer to review --
it never contacts the listing agent or schedules anything itself. See
`deployment/README.md` for why, and for the full AgentCore deployment path.

## Architecture

```
main.py (AgentCore entrypoint, agent-agnostic)
  -> agent_demo/platform/registry.py: load_agent()   # picks an AgentSpec via AGENT_ID
  -> agent_demo/platform/harness.py: run(agent, payload)
       -> agent_demo/platform/tracing.py             # Arize AX / OpenInference, always runs first
       -> agent_demo/platform/kill_switch.py          # checked before any other work, always
       -> agent.build_tools(resources)                 # agent-supplied, e.g.:
            agent_demo/agents/house_search/tools/
              postgres_tools.py                          # save/rate listings, draft viewing requests
              memory_tools.py                             # remember/recall facts + preferences
            agent_demo/agents/faq_agent/tools.py            # a single static-answer tool
       -> agent_demo/platform/llm.py                   # <agent's model names> -> fallback, via LLM gateway
            -> litellm proxy (:4000)                      # holds ANTHROPIC_API_KEY, routes to Anthropic
       -> agent_demo/platform/graph_factory.py         # StateGraph: agent <-> tools -> critic (shared by every agent)
       -> agent_demo/lib/memory/                        # shared infra, opt-in (not platform-enforced)
            short_term.py                                 # MongoDBSaver checkpointer (per session)
            long_term.py                                  # MongoLongTermStore
            factual.py                                     # FactualMemory (research notes)
```

An agent's entire surface is `agent_demo/platform/spec.py`'s `AgentSpec`
Protocol -- compare `agent_demo/agents/house_search/spec.py` (Postgres, MCP,
memory tools, human-in-the-loop) against `agent_demo/agents/faq_agent/spec.py`
(one static tool, nothing else) to see how little a new agent has to supply.

## Local development

Requires Python 3.13, `uv`, and Docker (for local Mongo/Postgres/the LLM
gateway).

```sh
cp .env.example .env        # fill in ANTHROPIC_API_KEY and LLM_GATEWAY_API_KEY at minimum
docker compose up -d        # local MongoDB + Postgres + LiteLLM gateway
uv sync

# Invoke an agent directly, no HTTP layer (defaults to house-search):
uv run python scripts/run_local.py "Find me a 3-bedroom house in Lisbon"
AGENT_ID=faq-agent uv run python scripts/run_local.py "What does the kill switch do?"

# Or serve the same /invocations + /ping contract AgentCore Runtime uses
# (AGENT_ID picks which registered agent this process serves; defaults to house-search):
uv run python -m main
curl -X POST localhost:8080/invocations -H 'content-type: application/json' -d '{
  "message": "Find me a 3-bedroom house in Lisbon",
  "buyer_id": "buyer_123",
  "buyer_profile": "Family of three, budget EUR 450k, wants a garden and good transit access"
}'
```

`max_react_steps` and `max_self_correction_retries` (each agent's own
defaults, e.g. `agent_demo/agents/house_search/config.py`) can also be
overridden per-request by adding either field, e.g. `"max_react_steps": 4`,
to the JSON body above -- useful for a "quick look" call that should stop
sooner than the default. Either way, the effective value is clamped to the
platform-wide `MAX_REACT_STEPS_CEILING`/`MAX_SELF_CORRECTION_RETRIES_CEILING`
(`agent_demo/platform/config.py`), which no agent or per-request override can
exceed.

Web search defaults to the reference Brave Search MCP server -- set
`BRAVE_API_KEY` in `.env`, or swap `MCP_WEB_SEARCH_COMMAND`/`MCP_WEB_SEARCH_ARGS`
for a different MCP-compatible provider (Exa, Tavily, ...). Tracing
(`ARIZE_SPACE_ID`/`ARIZE_API_KEY`) and the AgentCore deployment path are both
optional for local dev -- everything else runs against real Anthropic/Mongo/
Postgres/MCP services with just the steps above.

### LLM gateway

`agent_demo/platform/llm.py` never calls Anthropic directly -- it talks to a
LiteLLM proxy (`LLM_GATEWAY_BASE_URL`, default `http://localhost:4000`) over
the OpenAI-compatible API via `ChatOpenAI`, and authenticates with
`LLM_GATEWAY_API_KEY`. It's the only place in the codebase that constructs a
model client; `AgentSpec` has no hook for an agent to substitute one, so
every agent's calls go through the gateway unconditionally. The
`docker compose up -d` step above starts that proxy from
`litellm_config.yaml`, which is where `ANTHROPIC_API_KEY` and the
`PRIMARY_MODEL`/`FALLBACK_MODEL` -> real-model mapping live -- the app
process never sees the Anthropic key. To add a model or switch providers,
edit `litellm_config.yaml`'s `model_list` (and each agent's own
`primary_model`/`fallback_model` config), not `agent_demo/platform/llm.py`.
In production, point `LLM_GATEWAY_BASE_URL` at a centrally-hosted LiteLLM (or
other OpenAI-compatible) gateway instead of the local container.

## Testing

```sh
uv sync --group dev
uv run pytest
```

- `tests/platform/` covers the platform itself: the shared ReAct+critic graph
  factory (self-correction, retry give-up, step-budget grace turn) against a
  fake model and LangGraph's in-memory checkpointer/store; the kill switch
  against a fake Mongo collection; and `harness.run`'s enforcement
  guarantees (kill switch ordering, tracing-once, budget clamping,
  agent-independent `recursion_limit`) against a `NullAgent` test double that
  supplies no business logic at all.
- `tests/agents/` covers each agent's own tools/spec -- e.g. house-search's
  human-in-the-loop `draft_viewing_request` gate, run through the real
  platform graph factory against a fake Postgres pool.
- `tests/test_no_raw_interrupt.py` is a CI backstop: it fails if any agent
  tool calls LangGraph's `interrupt()` directly instead of going through
  `agent_demo/platform/hitl.py`'s `request_approval` -- the one seam that
  isn't structurally enforced (see that module's docstring).

## Known limitations

- **Prompt injection surface.** `buyer_profile` and the user's `message`
  (see `HouseSearchRequest` in `agent_demo/agents/house_search/spec.py`, and
  `BaseInvokeEnvelope` in `agent_demo/platform/envelope.py`) are untrusted
  text that flows directly into the system/human prompt of an agent with real side
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
