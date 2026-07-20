"""House-search agent config: business-specific settings (which models it
asks the gateway for, its own default budgets, its Postgres/MCP dependencies)
-- as opposed to deployment/operational settings, which live in
`agent_demo.platform.config`.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HouseSearchConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Models this agent asks the gateway for ---
    primary_model: str = Field(default="claude-opus-4-8", alias="PRIMARY_MODEL")
    fallback_model: str = Field(default="claude-haiku-4-5", alias="FALLBACK_MODEL")

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

    # --- Agent behavior defaults ---
    # Validated at registration time (agent_demo.platform.registry) to not
    # exceed the platform's max_react_steps_ceiling/
    # max_self_correction_retries_ceiling.
    default_max_react_steps: int = Field(default=12, alias="MAX_REACT_STEPS")
    default_max_self_correction_retries: int = Field(
        default=2, alias="MAX_SELF_CORRECTION_RETRIES"
    )


house_search_config = HouseSearchConfig()  # type: ignore[call-arg]
