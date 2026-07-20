"""`HouseSearchAgent`: the only thing this agent's author had to write to be
served by the platform harness. It supplies its own request schema, state
schema, system prompt renderer, and tool assembly -- and nothing else. It has
no access to the kill switch, tracing setup, or a raw model client; those are
owned entirely by `agent_demo.platform`.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pydantic import Field, model_validator

from agent_demo.agents.house_search.config import house_search_config
from agent_demo.agents.house_search.prompts import SYSTEM_PROMPT
from agent_demo.agents.house_search.state import HouseSearchState
from agent_demo.agents.house_search.tools.memory_tools import build_memory_tools
from agent_demo.agents.house_search.tools.postgres_tools import build_listing_tools, get_pool
from agent_demo.lib.mcp_tools import load_mcp_tools
from agent_demo.lib.memory.factual import FactualMemory
from agent_demo.platform.envelope import BaseInvokeEnvelope
from agent_demo.platform.spec import RequestResources


class HouseSearchRequest(BaseInvokeEnvelope):
    """`buyer_profile` is the one field that crosses a real trust boundary
    (arbitrary per-request text from the caller) -- it flows only into the
    prompt, never into a tool call or query, so no further sanitization is
    required beyond this length/type check.
    """

    buyer_id: str = Field(min_length=1, max_length=200)
    buyer_profile: str | None = Field(default=None, max_length=20000)

    @model_validator(mode="after")
    def _validate_buyer_profile_required_for_new_turns(self) -> "HouseSearchRequest":
        if self.resume_decision is None and not self.buyer_profile:
            raise ValueError(
                "buyer_profile is required unless resuming with resume_decision."
            )
        return self


# Process-wide MCP tool connections, cached like the Postgres pool in
# postgres_tools.py -- reconnecting per request would be wasteful and these
# tools carry no per-session state.
_mcp_tools: list[BaseTool] | None = None


async def _get_mcp_tools() -> list[BaseTool]:
    global _mcp_tools
    if _mcp_tools is None:
        _mcp_tools = await load_mcp_tools(
            web_search_command=house_search_config.mcp_web_search_command,
            web_search_args=house_search_config.mcp_web_search_args,
            brave_api_key=house_search_config.brave_api_key,
            fetch_command=house_search_config.mcp_fetch_command,
            fetch_args=house_search_config.mcp_fetch_args,
        )
    return _mcp_tools


class HouseSearchAgent:
    agent_id = "house-search"
    request_schema = HouseSearchRequest
    state_schema = HouseSearchState

    primary_model = house_search_config.primary_model
    fallback_model = house_search_config.fallback_model

    default_max_react_steps = house_search_config.default_max_react_steps
    default_max_self_correction_retries = house_search_config.default_max_self_correction_retries

    def build_initial_domain_state(self, request: HouseSearchRequest) -> dict:
        return {"buyer_id": request.buyer_id, "buyer_profile": request.buyer_profile}

    def render_system_prompt(self, state: HouseSearchState) -> str:
        return SYSTEM_PROMPT.format(buyer_profile=state["buyer_profile"])

    async def build_tools(self, resources: RequestResources) -> list[BaseTool]:
        request: HouseSearchRequest = resources.request
        factual_memory = FactualMemory(resources.mongo_client)
        pg_pool = await get_pool()
        return [
            *await _get_mcp_tools(),
            *build_listing_tools(pg_pool, resources.session_id),
            *build_memory_tools(factual_memory, resources.long_term_store, request.buyer_id),
        ]
