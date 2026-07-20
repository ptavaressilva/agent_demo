"""Graph state fields every agent gets for free via the ReAct+critic factory
graph (see graph_factory.py). An agent's own state TypedDict extends this
with whatever domain fields it needs (e.g. buyer_id/buyer_profile) -- those
extra fields are opaque to the factory graph, which only ever reads/writes
the fields declared here.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class BaseAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    # Fixed for the lifetime of a session (set once, at graph invocation).
    session_id: str

    # Budgets for this run -- already clamped to the platform ceiling by the
    # time they reach graph state (see platform.budget.clamp_budget).
    max_react_steps: int
    max_self_correction_retries: int

    # ReAct loop bookkeeping.
    react_steps: int
    correction_retries: int
    budget_stop_issued: bool
