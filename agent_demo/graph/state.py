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

    # ReAct loop bookkeeping.
    react_steps: int
    correction_retries: int
