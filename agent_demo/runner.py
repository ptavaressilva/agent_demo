"""Core agent invocation logic, independent of how it's hosted (AgentCore,
local CLI, tests). This is the part of the system that's fully testable
without deploying anything -- `main.py` is a thin AgentCore adapter around
`run()` below.

Process-wide resources (Mongo client, Postgres pool, MCP tool connections)
are created lazily and cached for the life of the process; only the parts
that vary per request (listing-persistence tools bound to `session_id`,
memory tools bound to `buyer_id`, and the graph's own conversation state) are
rebuilt per call.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from pymongo import MongoClient

from agent_demo.config import settings
from agent_demo.graph.graph import build_graph
from agent_demo.llm import build_chat_model_with_fallback
from agent_demo.memory.factual import FactualMemory
from agent_demo.memory.long_term import MongoLongTermStore
from agent_demo.memory.short_term import build_checkpointer
from agent_demo.tools.mcp_tools import load_mcp_tools
from agent_demo.tools.memory_tools import build_memory_tools
from agent_demo.tools.postgres_tools import build_listing_tools, get_pool
from agent_demo.tracing.setup import configure_tracing


class InvokeRequest(BaseModel):
    """Validated shape of an incoming invocation. `buyer_profile` is the
    one field that crosses a real trust boundary (arbitrary per-request text
    from the caller) -- it flows only into the prompt, never into a tool
    call or query, so no further sanitization is required beyond this
    length/type check.
    """

    message: str = Field(min_length=1, max_length=8000)
    buyer_id: str = Field(min_length=1, max_length=200)
    buyer_profile: str = Field(min_length=1, max_length=20000)
    session_id: str | None = None

    # Optional per-request overrides of the process-wide defaults in
    # `settings`, e.g. a "quick look" request that should stop sooner.
    max_react_steps: int | None = Field(default=None, ge=1, le=50)
    max_self_correction_retries: int | None = Field(default=None, ge=0, le=10)


@dataclass
class InvokeResult:
    session_id: str
    reply: str
    message_count: int


_NO_REPLY_FALLBACK = (
    "(No summary was produced for this turn -- the agent may have run out "
    "of steps mid-action. Try again or ask for a status update.)"
)


def _extract_reply(messages: list) -> str:
    """The last non-empty text content among the AIMessages in `messages`.

    The graph's step-budget/retry-limit grace turns are designed to always
    produce real summary text (see graph.py's `budget_stop` node and
    `GIVE_UP_PROMPT`), but that's a model-compliance guarantee, not a
    structural one -- a grace turn could still reply with only a
    (now-discarded) tool call and no text. Never surface a silent empty
    string to the caller in that case.
    """
    return next(
        (
            message.content
            for message in reversed(messages)
            if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content
        ),
        _NO_REPLY_FALLBACK,
    )


_mongo_client: MongoClient | None = None
_mcp_tools: list[BaseTool] | None = None


def _get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(settings.mongo_uri)
    return _mongo_client


async def _get_mcp_tools() -> list[BaseTool]:
    global _mcp_tools
    if _mcp_tools is None:
        _mcp_tools = await load_mcp_tools()
    return _mcp_tools


async def run(payload: dict) -> InvokeResult:
    """Handle one agent turn end-to-end: build the tool/model/graph stack
    for this request, run the ReAct + self-correction loop, and return the
    agent's reply."""
    configure_tracing()
    request = InvokeRequest.model_validate(payload)
    session_id = request.session_id or str(uuid.uuid4())

    mongo_client = _get_mongo_client()
    checkpointer = build_checkpointer(mongo_client)
    long_term_store = MongoLongTermStore(mongo_client)
    factual_memory = FactualMemory(mongo_client)
    pg_pool = await get_pool()

    tools: list[BaseTool] = [
        *await _get_mcp_tools(),
        *build_listing_tools(pg_pool, session_id),
        *build_memory_tools(factual_memory, long_term_store, request.buyer_id),
    ]

    model = await build_chat_model_with_fallback(tools)
    graph = build_graph(model, tools, checkpointer, long_term_store)

    graph_config = {"configurable": {"thread_id": session_id}}
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=request.message)],
            "session_id": session_id,
            "buyer_id": request.buyer_id,
            "buyer_profile": request.buyer_profile,
            "max_react_steps": request.max_react_steps or settings.max_react_steps,
            "max_self_correction_retries": (
                request.max_self_correction_retries
                if request.max_self_correction_retries is not None
                else settings.max_self_correction_retries
            ),
            "react_steps": 0,
            "correction_retries": 0,
            "budget_stop_issued": False,
        },
        config=graph_config,
    )

    reply = _extract_reply(result["messages"])
    return InvokeResult(session_id=session_id, reply=reply, message_count=len(result["messages"]))
