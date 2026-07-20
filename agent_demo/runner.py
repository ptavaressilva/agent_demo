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
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator
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

    message: str | None = Field(default=None, min_length=1, max_length=8000)
    buyer_id: str = Field(min_length=1, max_length=200)
    buyer_profile: str | None = Field(default=None, max_length=20000)
    session_id: str | None = None

    # Optional per-request overrides of the process-wide defaults in
    # `settings`, e.g. a "quick look" request that should stop sooner.
    max_react_steps: int | None = Field(default=None, ge=1, le=50)
    max_self_correction_retries: int | None = Field(default=None, ge=0, le=10)

    # Set to resume a run paused on a human-in-the-loop approval (see
    # draft_viewing_request's `interrupt()` call) instead of starting a new
    # turn -- e.g. {"action": "approve"} or {"action": "reject"}, optionally
    # with edited fields that override the tool's original arguments.
    # Crosses the same trust boundary as `buyer_profile`: arbitrary
    # per-request data that, on approval, flows straight into a tool's DB
    # write, so it gets the same length/type check and no more.
    resume_decision: dict[str, str] | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_message_xor_resume(self) -> "InvokeRequest":
        if self.resume_decision is not None:
            if not self.session_id:
                raise ValueError("resume_decision requires an existing session_id.")
            if any(len(value) > 20000 for value in self.resume_decision.values()):
                raise ValueError("resume_decision values must each be at most 20000 characters.")
        elif not self.message or not self.buyer_profile:
            raise ValueError("message and buyer_profile are required unless resuming with resume_decision.")
        return self


@dataclass
class InvokeResult:
    session_id: str
    reply: str
    message_count: int
    # Set when the run paused on a human-in-the-loop approval instead of
    # finishing -- the interrupt's payload (see draft_viewing_request).
    # Resume by calling `run()` again with the same session_id and a
    # `resume_decision`.
    pending_approval: dict | None = None


_NO_REPLY_FALLBACK = (
    "(No summary was produced for this turn -- the agent may have run out "
    "of steps mid-action. Try again or ask for a status update.)"
)

_PENDING_APPROVAL_REPLY = (
    "(Paused for approval -- see pending_approval. Resume by calling again "
    "with the same session_id and a resume_decision.)"
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


def _build_graph_input(request: InvokeRequest, session_id: str) -> dict | Command:
    """The input to feed `graph.ainvoke`: a `Command(resume=...)` if this
    call is responding to a pending human-in-the-loop approval, otherwise a
    fresh turn's initial state.

    A resumed run's state comes entirely from the checkpoint LangGraph
    already has for `session_id` -- never merge the two, or you'd be
    re-seeding react_steps/correction_retries/etc. alongside the resume
    value instead of continuing from where the interrupt paused.
    """
    if request.resume_decision is not None:
        return Command(resume=request.resume_decision)
    return {
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
    }


async def run(payload: dict) -> InvokeResult:
    """Handle one agent turn end-to-end: build the tool/model/graph stack
    for this request, run the ReAct + self-correction loop, and return the
    agent's reply -- or, if a tool (e.g. draft_viewing_request) paused for
    human approval, return that pending approval instead."""
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
    result = await graph.ainvoke(_build_graph_input(request, session_id), config=graph_config)

    if pending := result.get("__interrupt__"):
        return InvokeResult(
            session_id=session_id,
            reply=_PENDING_APPROVAL_REPLY,
            message_count=len(result.get("messages", [])),
            pending_approval=pending[0].value,
        )

    reply = _extract_reply(result["messages"])
    return InvokeResult(session_id=session_id, reply=reply, message_count=len(result["messages"]))
