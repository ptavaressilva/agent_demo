"""Graph state shared across all nodes."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    # Fixed for the lifetime of a session (set once, at graph invocation).
    session_id: str
    buyer_id: str
    buyer_profile: str

    # Budgets for this run. Default to `settings.max_react_steps` /
    # `settings.max_self_correction_retries`; a caller may override either
    # per-request (see `InvokeRequest`) for a "quick" vs. "thorough" run.
    max_react_steps: int
    max_self_correction_retries: int

    # ReAct loop bookkeeping.
    react_steps: int
    correction_retries: int
    budget_stop_issued: bool
