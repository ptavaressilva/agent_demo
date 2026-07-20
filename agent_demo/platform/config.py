"""Platform (deployment/operational) settings, loaded from environment
variables / .env. Owned by whoever operates the deployment, not by agent
authors -- agent-specific config (models, domain budgets, business-specific
integrations) lives instead in each `agent_demo/agents/<agent>/config.py`.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlatformSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM gateway ---
    # All model calls are routed through an LLM gateway (LiteLLM proxy --
    # see litellm_config.yaml) instead of hitting a provider directly. The
    # gateway holds the real provider credentials; this process only ever
    # sees a gateway key. In local dev the key comes straight from this env
    # var. When deployed to AgentCore Runtime, set
    # LLM_GATEWAY_AUTH_MODE=agentcore_identity instead -- the key is then
    # resolved per-call from an AgentCore Identity API-key credential
    # provider (see agent_demo/platform/llm.py and deployment/README.md) and
    # never needs to be baked into the container.
    llm_gateway_base_url: str = Field(
        default="http://localhost:4000", alias="LLM_GATEWAY_BASE_URL"
    )
    llm_gateway_api_key: str = Field(default="", alias="LLM_GATEWAY_API_KEY")
    llm_gateway_auth_mode: str = Field(default="env", alias="LLM_GATEWAY_AUTH_MODE")
    agentcore_model_provider_api_key_name: str = Field(
        default="llm-gateway-api-key", alias="BEDROCK_AGENTCORE_MODEL_PROVIDER_API_KEY_NAME"
    )

    # --- MongoDB (short-term / long-term / factual memory, kill switch) ---
    mongo_uri: str = Field(default="mongodb://localhost:27017", alias="MONGO_URI")
    mongo_db_name: str = Field(default="agent_demo", alias="MONGO_DB_NAME")

    # --- Arize AX tracing ---
    arize_space_id: str = Field(default="", alias="ARIZE_SPACE_ID")
    arize_api_key: str = Field(default="", alias="ARIZE_API_KEY")
    tracing_enabled: bool = Field(default=True, alias="TRACING_ENABLED")

    # --- Kill switch ---
    # Shared secret checked against the X-Admin-Token header on the
    # /admin/kill-switch endpoint (see main.py). Left empty by default so the
    # endpoint fails closed (rejects every request) until an operator
    # deliberately sets it, rather than defaulting to open.
    kill_switch_admin_token: str = Field(default="", alias="KILL_SWITCH_ADMIN_TOKEN")

    # --- Budget ceilings ---
    # Hard platform-wide caps no agent (or per-request override) can exceed,
    # regardless of what the agent's own default or a caller's override
    # requests. See agent_demo.platform.budget.clamp_budget and
    # agent_demo.platform.registry (which fails process startup if an
    # agent's own default exceeds these).
    max_react_steps_ceiling: int = Field(default=30, alias="MAX_REACT_STEPS_CEILING")
    max_self_correction_retries_ceiling: int = Field(
        default=5, alias="MAX_SELF_CORRECTION_RETRIES_CEILING"
    )
    # Multiplied by max_react_steps_ceiling to get the LangGraph
    # `recursion_limit` passed to every `graph.ainvoke` -- a structural
    # backstop independent of the per-request step counter, since a
    # ReAct+critic turn is several LangGraph super-steps, not one.
    recursion_limit_multiplier: int = Field(default=4, alias="RECURSION_LIMIT_MULTIPLIER")


platform_settings = PlatformSettings()  # type: ignore[call-arg]
