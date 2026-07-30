# How to add a new agent

This is a guide for onboarding a **new agent** onto this platform. It's
written from the perspective of an agent author — someone who wants their
own LangGraph agent hosted by this deployment — not from the perspective of
someone changing the platform itself.

The short version: you write one small package under `agent_demo/agents/`,
add one line to a manifest, and you're done. You never edit anything under
`agent_demo/platform/` or `main.py` — a CI test
(`tests/test_no_agent_names_in_platform.py`) fails the build if you try to.

## The contract you're implementing

Everything the platform needs from your agent is the `AgentSpec` Protocol in
[`agent_demo/platform/spec.py`](../agent_demo/platform/spec.py):

```python
class AgentSpec(Protocol):
    agent_id: str
    request_schema: type[BaseInvokeEnvelope]
    state_schema: type[BaseAgentState]
    primary_model: str
    fallback_model: str
    default_max_react_steps: int
    default_max_self_correction_retries: int

    def build_initial_domain_state(self, request) -> dict: ...
    def render_system_prompt(self, state) -> str: ...
    async def build_tools(self, resources: RequestResources) -> list[BaseTool]: ...
```

That's it. In exchange for implementing this, you get for free, on every
call, with no way to opt out or bypass:

- the operator **kill switch**, checked before anything else runs
- **Arize AX tracing** for every LLM/tool/graph call
- **LLM gateway routing** — you only ever name a model (e.g.
  `"claude-opus-4-8"`), never construct a client or hold a provider key
- a **ReAct + self-correction loop** (agent ↔ tools, with a critic node that
  catches failing tool calls and asks the model to correct course, bounded by
  retries) — or a different loop shape if you opt into one, see
  [Advanced: a different loop shape](#advanced-a-different-loop-shape)
- **step/retry budget clamping** to a platform-wide ceiling that no agent or
  caller can exceed

`agent_demo/agents/faq_agent/` is the smallest possible real example — copy
it as a starting template. `agent_demo/agents/house_search/` shows the same
contract carrying a Postgres pool, MCP tools, long-term memory, and a
human-in-the-loop approval.

## Step by step

### 1. Create your agent's package

```
agent_demo/agents/<your_agent>/
  __init__.py
  config.py
  state.py
  prompts.py
  tools.py
  spec.py
```

Nothing enforces this exact layout — `faq_agent`'s one-file-per-concern
split is a convention, not a requirement. What matters is that `spec.py`
ends up exporting a class implementing `AgentSpec`.

### 2. Config: model choice and budget defaults

Your agent's model names and step/retry budgets are a business choice, not a
platform one — they live in your own `config.py`, loaded from env vars:

```python
# agent_demo/agents/<your_agent>/config.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class YourAgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    primary_model: str = Field(default="claude-opus-4-8", alias="YOUR_AGENT_PRIMARY_MODEL")
    fallback_model: str = Field(default="claude-haiku-4-5", alias="YOUR_AGENT_FALLBACK_MODEL")

    default_max_react_steps: int = Field(default=4, alias="YOUR_AGENT_MAX_REACT_STEPS")
    default_max_self_correction_retries: int = Field(
        default=1, alias="YOUR_AGENT_MAX_SELF_CORRECTION_RETRIES"
    )


your_agent_config = YourAgentConfig()  # type: ignore[call-arg]
```

`primary_model`/`fallback_model` are names the LLM gateway resolves
(`litellm_config.yaml`) — you never see a provider API key. Keep
`default_max_react_steps`/`default_max_self_correction_retries` at or under
the platform ceilings
(`MAX_REACT_STEPS_CEILING`/`MAX_SELF_CORRECTION_RETRIES_CEILING` in
`agent_demo/platform/config.py`, default 30/5) — if you set a default above
the ceiling, the platform refuses to start the process rather than silently
clamping it later.

### 3. State: what persists across a session

Extend `BaseAgentState` with whatever domain fields your agent needs to
remember for the life of a session (a checkpointer persists this across
turns and human-in-the-loop resumes):

```python
# agent_demo/agents/<your_agent>/state.py
from agent_demo.platform.state import BaseAgentState


class YourAgentState(BaseAgentState):
    topic: str | None  # whatever your agent needs, fixed for the session
```

Any field you write in `build_initial_domain_state` (step 6) must be
declared here, or LangGraph will silently drop it.

### 4. Request: what a caller sends you

Extend `BaseInvokeEnvelope`, which already gives you `message`,
`session_id`, budget overrides, and `resume_decision` (for resuming a
paused human-in-the-loop approval):

```python
# in your spec.py, or a separate schema module
from pydantic import Field
from agent_demo.platform.envelope import BaseInvokeEnvelope


class YourAgentRequest(BaseInvokeEnvelope):
    topic: str | None = Field(default=None, max_length=200)
```

Any field here is untrusted, caller-supplied text — apply the same
length/type discipline `message` already gets (see `HouseSearchRequest` in
`agent_demo/agents/house_search/spec.py` for an example with a required
field and a custom validator).

### 5. Prompt

A plain string (or a function of state, if you need per-session
interpolation — see `HouseSearchAgent.render_system_prompt`, which formats
in the buyer profile):

```python
# agent_demo/agents/<your_agent>/prompts.py
SYSTEM_PROMPT = """You are ... Use the <your_tool> tool to ..."""
```

### 6. Tools

Plain LangChain `@tool`-decorated functions. One convention to follow: a
tool that fails should either raise, or return a string starting with
`"Error: "` — the platform's critic node looks for both signals to trigger
self-correction. If your tool "returns, don't raises" on failure and skips
the `"Error: "` prefix, the critic won't see it.

```python
# agent_demo/agents/<your_agent>/tools.py
from langchain_core.tools import BaseTool, tool


def build_your_tools() -> list[BaseTool]:
    @tool
    async def do_the_thing(arg: str) -> str:
        """Docstring becomes the tool description the model sees."""
        if not arg:
            return "Error: arg is required."
        return f"did the thing with {arg}"

    return [do_the_thing]
```

If your tool needs a real side effect a human should approve first (writing
somewhere, contacting someone, spending money), wrap it with
`agent_demo/platform/hitl.py`'s `request_approval`/`is_approved` instead of
calling LangGraph's `interrupt()` directly — see
`agent_demo/agents/house_search/tools/postgres_tools.py`'s
`draft_viewing_request` for a full example. A CI test
(`tests/test_no_raw_interrupt.py`) fails the build on a raw `interrupt(`
call outside that module.

### 7. The spec: wire it together

```python
# agent_demo/agents/<your_agent>/spec.py
from langchain_core.tools import BaseTool

from agent_demo.agents.<your_agent>.config import your_agent_config
from agent_demo.agents.<your_agent>.prompts import SYSTEM_PROMPT
from agent_demo.agents.<your_agent>.state import YourAgentState
from agent_demo.agents.<your_agent>.tools import build_your_tools
from agent_demo.platform.envelope import BaseInvokeEnvelope
from agent_demo.platform.spec import RequestResources
from pydantic import Field


class YourAgentRequest(BaseInvokeEnvelope):
    topic: str | None = Field(default=None, max_length=200)


class YourAgent:
    agent_id = "your-agent"
    request_schema = YourAgentRequest
    state_schema = YourAgentState

    primary_model = your_agent_config.primary_model
    fallback_model = your_agent_config.fallback_model

    default_max_react_steps = your_agent_config.default_max_react_steps
    default_max_self_correction_retries = your_agent_config.default_max_self_correction_retries

    def build_initial_domain_state(self, request: YourAgentRequest) -> dict:
        return {"topic": request.topic}

    def render_system_prompt(self, state: YourAgentState) -> str:
        return SYSTEM_PROMPT

    async def build_tools(self, resources: RequestResources) -> list[BaseTool]:
        return build_your_tools()
```

`resources` (a `RequestResources`) gives `build_tools` the process-wide
Mongo client, this request's validated payload, this session's id, and the
long-term store — not a chat model or a graph; those stay owned by the
platform. If your tools need their own backing store (a Postgres pool, an
MCP server connection), build/cache it the way
`agent_demo/agents/house_search/spec.py` does (`get_pool()`,
`_get_mcp_tools()`) rather than reconnecting on every call.

### 8. Register it

The only file outside your own package you touch:

```python
# agent_demo/agents/manifest.py
from agent_demo.agents.your_agent.spec import YourAgent

AGENTS: list[AgentSpec] = [HouseSearchAgent(), FaqAgent(), YourAgent()]
```

At process startup, `agent_demo/platform/registry.py` validates your
agent's budget defaults against the platform ceilings and refuses to start
if they're too high — better to catch that at deploy time than have every
request silently clamped.

### 9. Run it

```sh
AGENT_ID=your-agent uv run python scripts/run_local.py "hello"

# or serve it over HTTP like AgentCore does:
AGENT_ID=your-agent uv run python -m main
curl -X POST localhost:8080/invocations -H 'content-type: application/json' \
  -d '{"message": "hello"}'
```

Set `AGENT_ID` as the default for the whole deployment by editing
`DEFAULT_AGENT_ID` in `agent_demo/agents/manifest.py`, or leave it and pick
the agent per-request/per-process via the env var.

### 10. Test it

Follow `tests/agents/faq_agent/test_faq_agent.py` or
`tests/agents/house_search/` as a template: exercise your tools directly,
and/or run your `AgentSpec` through the real platform graph factory with a
fake model, the way `tests/platform/test_harness.py`'s `NullAgent` proves
the kill-switch/tracing/budget guarantees hold for *any* spec.

## Advanced: a different loop shape

The default ReAct + self-correction loop
(`agent_demo/platform/graph_factory.py`) fits most agents. If yours needs
different control flow — e.g. plan the whole task up front, then execute
step by step — set a class attribute on your spec:

```python
from agent_demo.platform.plan_execute_graph import build_plan_execute_graph, PlanExecuteState

class YourAgent:
    ...
    state_schema = PlanExecuteState  # or an extension of it
    graph_factory = build_plan_execute_graph
```

`harness.run` looks this up via `getattr(type(agent), "graph_factory",
None)` and falls back to the default when it's absent — kill switch,
tracing, gateway routing, and budget clamping are enforced identically
either way, because those live in `harness.run`, not in the graph factory.
What you *don't* automatically get from a different factory: the default's
self-correction critic and step-budget grace turn are specific to
`build_react_graph` — see `agent_demo/platform/plan_execute_graph.py`'s
module docstring for the tradeoffs of the plan-and-execute example it
ships, and build your own equivalents if you want those guarantees back.

## What you should never need to touch

- `agent_demo/platform/*.py` — kill switch, tracing, LLM gateway, graph
  factory, budget clamping. If a change here feels necessary to onboard your
  agent, that's a sign the `AgentSpec` contract is missing something —
  raise it rather than reaching around it.
- `main.py` — it only ever calls `registry.load_agent()`; it has no
  per-agent logic to add.

If you find yourself importing from `agent_demo.platform` internals beyond
`AgentSpec`/`RequestResources`/`BaseAgentState`/`BaseInvokeEnvelope`/
`hitl.py`, or importing your agent's own package from anywhere under
`agent_demo/platform/`, you've stepped outside the intended seam — see
`tests/test_no_agent_names_in_platform.py`, which enforces the latter in CI.
