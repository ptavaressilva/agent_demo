from __future__ import annotations

from agent_demo.platform.state import BaseAgentState


class FaqAgentState(BaseAgentState):
    # Fixed for the lifetime of a session (set once, at graph invocation).
    topic: str | None
