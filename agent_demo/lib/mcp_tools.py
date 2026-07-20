"""Web search and page-fetch tools, sourced from MCP servers over stdio.

Any MCP-compliant server works for each slot -- an agent picks the
command/args (which servers to talk to at all is a business choice, e.g.
whether an agent needs web search) and passes them in here; this module has
no config of its own to keep it reusable across agents that may want
different servers, or no MCP tools at all.

Each server is connected to independently: if one is unreachable or
misconfigured (e.g. a missing API key), it's skipped with a logged warning
rather than taking down tool loading for the other server too.
"""

from __future__ import annotations

import logging
import shlex

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


def _mcp_client(
    *,
    web_search_command: str,
    web_search_args: str,
    brave_api_key: str,
    fetch_command: str,
    fetch_args: str,
) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "web_search": {
                "transport": "stdio",
                "command": web_search_command,
                "args": shlex.split(web_search_args),
                "env": {"BRAVE_API_KEY": brave_api_key} if brave_api_key else None,
            },
            "web_fetch": {
                "transport": "stdio",
                "command": fetch_command,
                "args": shlex.split(fetch_args),
            },
        }
    )


async def load_mcp_tools(
    *,
    web_search_command: str,
    web_search_args: str,
    brave_api_key: str = "",
    fetch_command: str,
    fetch_args: str,
) -> list[BaseTool]:
    """Connect to the configured MCP servers and return their tools as
    LangChain `BaseTool`s, ready to bind to the model. Servers are loaded
    independently so a single unreachable/misconfigured server degrades
    gracefully instead of preventing the agent from starting at all.
    """
    client = _mcp_client(
        web_search_command=web_search_command,
        web_search_args=web_search_args,
        brave_api_key=brave_api_key,
        fetch_command=fetch_command,
        fetch_args=fetch_args,
    )
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
