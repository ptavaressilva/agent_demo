"""Config for the FAQ agent -- a deliberately minimal second agent with no
Postgres, no MCP servers, and no human-in-the-loop tool. It exists to prove
that onboarding a new agent onto the platform harness requires touching
nothing under `agent_demo/platform/` or `main.py`.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FaqAgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    primary_model: str = Field(default="claude-opus-4-8", alias="FAQ_AGENT_PRIMARY_MODEL")
    fallback_model: str = Field(default="claude-haiku-4-5", alias="FAQ_AGENT_FALLBACK_MODEL")

    default_max_react_steps: int = Field(default=4, alias="FAQ_AGENT_MAX_REACT_STEPS")
    default_max_self_correction_retries: int = Field(
        default=1, alias="FAQ_AGENT_MAX_SELF_CORRECTION_RETRIES"
    )


faq_agent_config = FaqAgentConfig()  # type: ignore[call-arg]
