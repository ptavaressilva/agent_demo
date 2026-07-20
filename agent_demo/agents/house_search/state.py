"""House-search's own graph state: the platform's base ReAct/budget bookkeeping
plus the domain fields this agent's prompt/tools need. Declared here (not on
`BaseAgentState`) so the platform's graph factory has no notion of what a
"buyer" is."""

from __future__ import annotations

from agent_demo.platform.state import BaseAgentState


class HouseSearchState(BaseAgentState):
    # Fixed for the lifetime of a session (set once, at graph invocation).
    buyer_id: str
    buyer_profile: str
