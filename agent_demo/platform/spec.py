"""`AgentSpec` is the entire surface an agent author implements. `harness.run`
owns the entrypoint and calls into an `AgentSpec` implementation only through
this fixed interface -- there is deliberately no hook here for an agent to
construct its own model client, or reach the kill switch/tracing/Mongo client
directly, so there is nothing for an agent author to forget or bypass: kill
switch, tracing, gateway routing, and budget ceilings are enforced by
`harness.run` itself for every `AgentSpec`, unconditionally.

The one deliberate exception is `graph_factory` below: an agent needing a
loop pattern other than the platform default (plain ReAct + self-correction
critic, see `platform.graph_factory.build_react_graph`) can supply its own,
e.g. `platform.plan_execute_graph.build_plan_execute_graph` for a
Plan-and-Execute loop. It's optional and deliberately *not* declared as a
Protocol member -- most agents never set it, and declaring it as a required
attribute would force every existing `AgentSpec` to redeclare a "just use the
default" no-op. `harness.run` looks it up with `getattr(type(agent),
"graph_factory", None)`, falling back to `build_react_graph` when absent.
Whichever factory runs, kill switch/tracing/gateway routing/budget
clamping/the `recursion_limit` backstop stay enforced identically -- those
live in `harness.run`, not in the factory. What does *not* carry over
automatically is `build_react_graph`'s own guarantees (the self-correction
critic, the graceful step-budget stop) -- an alternate factory has to build
its own equivalents if it wants them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from pymongo import MongoClient

from agent_demo.platform.envelope import BaseInvokeEnvelope
from agent_demo.platform.state import BaseAgentState


@dataclass
class RequestResources:
    """What `harness.run` hands an agent's `build_tools` to build this
    request's tools from -- process-wide resources plus this request's own
    validated payload. Notably absent: a chat model or a graph -- those stay
    owned by `platform.llm`/`platform.graph_factory`."""

    mongo_client: MongoClient
    session_id: str
    request: BaseInvokeEnvelope
    long_term_store: BaseStore


class AgentSpec(Protocol):
    agent_id: str
    request_schema: type[BaseInvokeEnvelope]
    state_schema: type[BaseAgentState]

    # Business choice, not a deployment one -- which models this agent asks
    # the gateway for. The gateway itself is not optional (see platform.llm).
    primary_model: str
    fallback_model: str

    default_max_react_steps: int
    default_max_self_correction_retries: int

    def build_initial_domain_state(self, request: BaseInvokeEnvelope) -> dict: ...

    def render_system_prompt(self, state: BaseAgentState) -> str: ...

    async def build_tools(self, resources: RequestResources) -> list[BaseTool]: ...

    # Optional, not a Protocol member on purpose (see module docstring): set
    # this class attribute to swap the platform's default ReAct+critic loop
    # for a different one, e.g.
    #
    #   from agent_demo.platform.plan_execute_graph import build_plan_execute_graph
    #   graph_factory = build_plan_execute_graph
    #
    # A plain function assigned here is safe to call unbound (no implicit
    # `self`) because `harness.run` looks it up via `type(agent)`, not
    # `agent` -- accessing a function through the class itself never binds it
    # to an instance the way `agent.graph_factory` would. Must match
    # `build_react_graph`'s call convention: `(model, tools, checkpointer,
    # store, system_prompt_fn=..., state_schema=...) -> CompiledStateGraph`.
