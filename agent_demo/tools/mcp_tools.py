"""Web search and page-fetch tools, sourced from MCP servers over stdio.

Any MCP-compliant server works for each slot -- swap the command/args in
config (`MCP_WEB_SEARCH_*` / `MCP_FETCH_*`) to point at a different provider
without touching agent code. Defaults to the reference Brave Search server
for search and the reference `mcp-server-fetch` for page retrieval.

Each server is connected to independently: if one is unreachable or
misconfigured (e.g. a missing API key), it's skipped with a logged warning
rather than taking down tool loading for the other server too.
"""

from __future__ import annotations

import logging
import shlex

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent_demo.config import settings

logger = logging.getLogger(__name__)


def _mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "web_search": {
                "transport": "stdio",
                "command": settings.mcp_web_search_command,
                "args": shlex.split(settings.mcp_web_search_args),
                "env": {"BRAVE_API_KEY": settings.brave_api_key} if settings.brave_api_key else None,
            },
            "web_fetch": {
                "transport": "stdio",
                "command": settings.mcp_fetch_command,
                "args": shlex.split(settings.mcp_fetch_args),
            },
        }
    )


async def load_mcp_tools() -> list[BaseTool]:
    """Connect to the configured MCP servers and return their tools as
    LangChain `BaseTool`s, ready to bind to the model. Servers are loaded
    independently so a single unreachable/misconfigured server degrades
    gracefully instead of preventing the agent from starting at all.
    """
    client = _mcp_client()
    tools: list[BaseTool] = []
    for server_name in ("web_search", "web_fetch"):
        try:
            tools.extend(await client.get_tools(server_name=server_name))
        except Exception:
            logger.warning(
                "Could not load tools from MCP server %r; continuing without it. "
                "Check its command/args/API key in config.",
                server_name,
                exc_info=True,
            )
    return tools
