"""Turns `agent_demo.agents.manifest.AGENTS` -- the list of agents this
deployment can serve, contributed from the agents' own namespace -- into a
lookup by `agent_id`. `main.py` selects one via the `AGENT_ID` env var and
never imports anything from `agent_demo.agents.*` beyond this lookup.

This module is the one place under `agent_demo/platform/` allowed to import
from `agent_demo.agents.*`, and even here only from the manifest -- never a
specific agent's class or package. Onboarding a new agent means adding it to
`agent_demo/agents/manifest.py`; nothing under `agent_demo/platform/` or
`main.py` changes (see tests/test_no_agent_names_in_platform.py, which
enforces this at CI time).

Registration validates each agent's own budget defaults against the
platform's ceilings (`platform.config.platform_settings`) and fails process
startup if an agent is misconfigured to exceed them -- catching that at
deploy time rather than relying on the per-request clamp in `harness.run` to
quietly mask it forever.
"""

from __future__ import annotations

import os

from agent_demo.agents.manifest import AGENTS, DEFAULT_AGENT_ID
from agent_demo.platform.config import platform_settings
from agent_demo.platform.spec import AgentSpec


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
    for agent in AGENTS:
        _validate_ceilings(agent)
    return {agent.agent_id: agent for agent in AGENTS}


_REGISTRY: dict[str, AgentSpec] = _build_registry()


def load_agent(agent_id: str | None = None) -> AgentSpec:
    agent_id = agent_id or os.environ.get("AGENT_ID", DEFAULT_AGENT_ID)
    try:
        return _REGISTRY[agent_id]
    except KeyError:
        raise RuntimeError(
            f"Unknown AGENT_ID {agent_id!r}. Registered agents: {sorted(_REGISTRY)}"
        ) from None
