"""The list of agents this deployment can serve, contributed from the
agents' own namespace. `agent_demo.platform.registry` consumes `AGENTS` (and
validates it against the platform's ceilings) but never itself names an
agent class or import path -- so onboarding a new agent means adding it
here, not to anything under `agent_demo/platform/` or `main.py`.
"""

from __future__ import annotations

from agent_demo.agents.faq_agent.spec import FaqAgent
from agent_demo.agents.house_search.spec import HouseSearchAgent
from agent_demo.platform.spec import AgentSpec

AGENTS: list[AgentSpec] = [HouseSearchAgent(), FaqAgent()]

# Which registered agent `main.py`/`scripts/run_local.py` serve when no
# `AGENT_ID` env var is set.
DEFAULT_AGENT_ID = "house-search"
