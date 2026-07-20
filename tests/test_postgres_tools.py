"""Tests for postgres_tools.draft_viewing_request's human-in-the-loop
approval gate: the DB write only happens after an explicit "approve" resume,
reviewer edits override the original draft, and a reject never touches the
database. Runs through the real graph (with an in-memory checkpointer) since
`interrupt()`/`Command(resume=...)` need actual LangGraph persistence -- but
against a fake Postgres pool, so no real database is needed.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from agent_demo.graph.graph import build_graph
from agent_demo.tools.postgres_tools import build_listing_tools


class FakeModel:
    """Returns each of `responses` in order, one per `ainvoke` call."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self.calls = 0

    async def ainvoke(self, _messages, *args, **kwargs) -> AIMessage:
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[index]


class _FakeConn:
    def __init__(self, executed: list) -> None:
        self._executed = executed

    async def execute(self, query: str, *args) -> None:
        self._executed.append((query, args))


class _FakeAcquire:
    def __init__(self, executed: list) -> None:
        self._executed = executed

    async def __aenter__(self) -> _FakeConn:
        return _FakeConn(self._executed)

    async def __aexit__(self, *exc) -> bool:
        return False


class FakePool:
    """Records every `conn.execute(...)` call made through `pool.acquire()`."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self.executed)


def _initial_state() -> dict:
    return {
        "messages": [HumanMessage(content="draft a viewing request")],
        "session_id": "s1",
        "buyer_id": "b1",
        "buyer_profile": "Family of three, budget EUR 450k",
        "max_react_steps": 12,
        "max_self_correction_retries": 2,
        "react_steps": 0,
        "correction_retries": 0,
        "budget_stop_issued": False,
    }


def _draft_tool_call(listing_id: str, inquiry_message: str, buyer_highlights: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "draft_viewing_request",
                "args": {
                    "listing_id": listing_id,
                    "inquiry_message": inquiry_message,
                    "buyer_highlights": buyer_highlights,
                },
                "id": "call_1",
            }
        ],
    )


async def test_draft_viewing_request_writes_only_after_approval():
    pool = FakePool()
    draft_tool = next(
        t for t in build_listing_tools(pool, session_id="s1") if t.name == "draft_viewing_request"
    )
    model = FakeModel(
        [
            _draft_tool_call("42", "Can we view this Saturday?", "Pre-approved, flexible move-in."),
            AIMessage(content="Drafted and saved."),
        ]
    )
    graph = build_graph(model, [draft_tool], InMemorySaver(), InMemoryStore())
    config = {"configurable": {"thread_id": "draft-1"}}

    paused = await graph.ainvoke(_initial_state(), config=config)

    interrupt_value = paused["__interrupt__"][0].value
    assert interrupt_value["action"] == "draft_viewing_request"
    assert interrupt_value["listing_id"] == "42"
    assert pool.executed == []

    await graph.ainvoke(Command(resume={"action": "approve"}), config=config)

    assert len(pool.executed) == 1
    _query, args = pool.executed[0]
    assert args == (42, "Can we view this Saturday?", "Pre-approved, flexible move-in.", "")


async def test_draft_viewing_request_approval_edits_override_the_draft():
    pool = FakePool()
    draft_tool = next(
        t for t in build_listing_tools(pool, session_id="s1") if t.name == "draft_viewing_request"
    )
    model = FakeModel(
        [
            _draft_tool_call("7", "original message", "original highlights"),
            AIMessage(content="Saved with your edits."),
        ]
    )
    graph = build_graph(model, [draft_tool], InMemorySaver(), InMemoryStore())
    config = {"configurable": {"thread_id": "draft-2"}}

    await graph.ainvoke(_initial_state(), config=config)
    await graph.ainvoke(
        Command(resume={"action": "approve", "inquiry_message": "edited message"}),
        config=config,
    )

    _query, args = pool.executed[0]
    assert args[1] == "edited message"
    assert args[2] == "original highlights"


async def test_draft_viewing_request_rejection_never_writes():
    pool = FakePool()
    draft_tool = next(
        t for t in build_listing_tools(pool, session_id="s1") if t.name == "draft_viewing_request"
    )
    model = FakeModel(
        [
            _draft_tool_call("9", "hello", "highlights"),
            AIMessage(content="Okay, I won't save that draft."),
        ]
    )
    graph = build_graph(model, [draft_tool], InMemorySaver(), InMemoryStore())
    config = {"configurable": {"thread_id": "draft-3"}}

    await graph.ainvoke(_initial_state(), config=config)
    result = await graph.ainvoke(Command(resume={"action": "reject"}), config=config)

    assert pool.executed == []
    assert result["messages"][-1].content == "Okay, I won't save that draft."
