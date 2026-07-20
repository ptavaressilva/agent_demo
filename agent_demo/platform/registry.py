"""Every agent this deployment can serve, keyed by `agent_id`. `main.py`
selects one via the `AGENT_ID` env var and never imports anything from
`agent_demo.agents.*` beyond this lookup -- adding a new agent means adding
it to `_build_registry` below, not touching `main.py` or `platform/`.

Registration validates each agent's own budget defaults against the
platform's ceilings (`platform.config.platform_settings`) and fails process
startup if an agent is misconfigured to exceed them -- catching that at
deploy time rather than relying on the per-request clamp in `harness.run` to
quietly mask it forever.
"""

from __future__ import annotations

import os

from agent_demo.platform.config import platform_settings
from agent_demo.platform.spec import AgentSpec

DEFAULT_AGENT_ID = "house-search"


def _validate_ceilings(agent: AgentSpec) -> None:
    if agent.default_max_react_steps > platform_settings.max_react_steps_ceiling:
        raise RuntimeError(
            f"Agent {agent.agent_id!r}: default_max_react_steps "
            f"({agent.default_max_react_steps}) exceeds the platform ceiling "
            f"({platform_settings.max_react_steps_ceiling})."
        )
    if (
        agent.default_max_self_correction_retries
        > platform_settings.max_self_correction_retries_ceiling
    ):
        raise RuntimeError(
            f"Agent {agent.agent_id!r}: default_max_self_correction_retries "
            f"({agent.default_max_self_correction_retries}) exceeds the platform "
            f"ceiling ({platform_settings.max_self_correction_retries_ceiling})."
        )


def _build_registry() -> dict[str, AgentSpec]:
    from agent_demo.agents.faq_agent.spec import FaqAgent
    from agent_demo.agents.house_search.spec import HouseSearchAgent

    agents: list[AgentSpec] = [HouseSearchAgent(), FaqAgent()]
    for agent in agents:
        _validate_ceilings(agent)
    return {agent.agent_id: agent for agent in agents}


AGENTS: dict[str, AgentSpec] = _build_registry()


def load_agent(agent_id: str | None = None) -> AgentSpec:
    agent_id = agent_id or os.environ.get("AGENT_ID", DEFAULT_AGENT_ID)
    try:
        return AGENTS[agent_id]
    except KeyError:
        raise RuntimeError(
            f"Unknown AGENT_ID {agent_id!r}. Registered agents: {sorted(AGENTS)}"
        ) from None
