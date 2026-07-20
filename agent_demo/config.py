"""Central settings, loaded from environment variables / .env."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Anthropic ---
    # In local dev the key comes straight from this env var. When deployed to
    # AgentCore Runtime, set ANTHROPIC_AUTH_MODE=agentcore_identity instead --
    # the key is then resolved per-call from an AgentCore Identity API-key
    # credential provider (see agent_demo/llm.py and deployment/README.md)
    # and never needs to be baked into the container.
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_auth_mode: str = Field(default="env", alias="ANTHROPIC_AUTH_MODE")
    agentcore_model_provider_api_key_name: str = Field(
        default="anthropic-api-key", alias="BEDROCK_AGENTCORE_MODEL_PROVIDER_API_KEY_NAME"
    )
    primary_model: str = Field(default="claude-opus-4-8", alias="PRIMARY_MODEL")
    fallback_model: str = Field(default="claude-haiku-4-5", alias="FALLBACK_MODEL")

    # --- MongoDB (short-term / long-term / factual memory) ---
    mongo_uri: str = Field(default="mongodb://localhost:27017", alias="MONGO_URI")
    mongo_db_name: str = Field(default="agent_demo", alias="MONGO_DB_NAME")

    # --- Postgres (house listings + ratings + viewing requests) ---
    postgres_dsn: str = Field(
        default="postgresql://agent_demo:agent_demo@localhost:5432/agent_demo",
        alias="POSTGRES_DSN",
    )

    # --- MCP servers ---
    # Any MCP-compatible web search server works (Brave, Exa, Tavily, ...) as
    # long as it's launched over stdio with this command/args. Swap the
    # command below for your provider's MCP server entrypoint.
    mcp_web_search_command: str = Field(default="npx", alias="MCP_WEB_SEARCH_COMMAND")
    mcp_web_search_args: str = Field(
        default="-y @modelcontextprotocol/server-brave-search",
        alias="MCP_WEB_SEARCH_ARGS",
    )
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")

    mcp_fetch_command: str = Field(default="uvx", alias="MCP_FETCH_COMMAND")
    mcp_fetch_args: str = Field(default="mcp-server-fetch", alias="MCP_FETCH_ARGS")

    # --- Arize AX tracing ---
    arize_space_id: str = Field(default="", alias="ARIZE_SPACE_ID")
    arize_api_key: str = Field(default="", alias="ARIZE_API_KEY")
    arize_project_name: str = Field(default="house-search-agent", alias="ARIZE_PROJECT_NAME")
    tracing_enabled: bool = Field(default=True, alias="TRACING_ENABLED")

    # --- Agent behavior ---
    max_react_steps: int = Field(default=12, alias="MAX_REACT_STEPS")
    max_self_correction_retries: int = Field(default=2, alias="MAX_SELF_CORRECTION_RETRIES")

    # --- Kill switch ---
    # Shared secret checked against the X-Admin-Token header on the
    # /admin/kill-switch endpoint (see main.py). Left empty by default so the
    # endpoint fails closed (rejects every request) until an operator
    # deliberately sets it, rather than defaulting to open.
    kill_switch_admin_token: str = Field(default="", alias="KILL_SWITCH_ADMIN_TOKEN")


settings = Settings()  # type: ignore[call-arg]
