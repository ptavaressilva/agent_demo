"""`AgentSpec` is the entire surface an agent author implements. `harness.run`
owns the entrypoint and calls into an `AgentSpec` implementation only through
this fixed interface -- there is deliberately no hook here for an agent to
construct its own model client, its own graph, or reach the kill
switch/tracing/Mongo client directly, so there is nothing for an agent author
to forget or bypass: kill switch, tracing, gateway routing, and budget
ceilings are enforced by `harness.run` itself for every `AgentSpec`,
unconditionally.
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
