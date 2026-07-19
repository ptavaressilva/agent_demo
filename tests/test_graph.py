"""Tests for the graph's control flow: the plain tool-use loop, the
self-correction critic (retry, then give up), and the step-budget grace
turn. All run against a scripted fake model and LangGraph's in-memory
checkpointer/store -- no Mongo/Postgres/MCP/Anthropic services needed, which
is exactly the point of keeping `agent_demo/graph/graph.py` free of any
direct dependency on those.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agent_demo.graph.graph import build_graph
from agent_demo.graph.prompts import CORRECTION_PROMPT, GIVE_UP_PROMPT, STEP_BUDGET_PROMPT


class FakeModel:
    """Returns each of `responses` in order, one per `ainvoke` call; the
    last response repeats if the graph calls more times than scripted."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self.calls = 0

    async def ainvoke(self, _messages, *args, **kwargs) -> AIMessage:
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[index]


@tool
async def good_tool(x: str) -> str:
    """A tool that always succeeds."""
    return f"ok:{x}"


@tool
async def bad_tool(x: str) -> str:
    """A tool that always fails, using this repo's error-string convention."""
    return "Error: always fails"


def _initial_state(
    *, max_react_steps: int = 12, max_self_correction_retries: int = 2
) -> dict:
    return {
        "messages": [HumanMessage(content="find me a house")],
        "session_id": "s1",
        "buyer_id": "b1",
        "buyer_profile": "Family of three, budget EUR 450k",
        "max_react_steps": max_react_steps,
        "max_self_correction_retries": max_self_correction_retries,
        "react_steps": 0,
        "correction_retries": 0,
        "budget_stop_issued": False,
    }


async def test_normal_tool_call_flow_ends_on_final_text():
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "good_tool", "args": {"x": "1"}, "id": "call_1"}],
            ),
            AIMessage(content="All done, found a great house."),
        ]
    )
    graph = build_graph(model, [good_tool], InMemorySaver(), InMemoryStore())

    result = await graph.ainvoke(
        _initial_state(), config={"configurable": {"thread_id": "t1"}}
    )

    assert result["messages"][-1].content == "All done, found a great house."
    assert result["react_steps"] == 2
    assert any(isinstance(m, ToolMessage) and m.content == "ok:1" for m in result["messages"])


async def test_self_correction_retries_then_succeeds():
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "bad_tool", "args": {"x": "1"}, "id": "call_1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "good_tool", "args": {"x": "1"}, "id": "call_2"}],
            ),
            AIMessage(content="Fixed it and found a listing."),
        ]
    )
    graph = build_graph(model, [bad_tool, good_tool], InMemorySaver(), InMemoryStore())

    result = await graph.ainvoke(
        _initial_state(), config={"configurable": {"thread_id": "t2"}}
    )

    assert result["messages"][-1].content == "Fixed it and found a listing."
    # Retry counter reset to 0 once the second (successful) tool call's
    # critic pass finds no errors.
    assert result["correction_retries"] == 0
    assert any(
        isinstance(m, HumanMessage) and m.content.startswith(CORRECTION_PROMPT.split("\n")[0])
        for m in result["messages"]
    )


async def test_self_correction_gives_up_after_retry_limit():
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "bad_tool", "args": {"x": "1"}, "id": "call_1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "bad_tool", "args": {"x": "1"}, "id": "call_2"}],
            ),
            AIMessage(content="Sorry, that failed both times."),
        ]
    )
    graph = build_graph(model, [bad_tool], InMemorySaver(), InMemoryStore())

    result = await graph.ainvoke(
        _initial_state(max_self_correction_retries=1),
        config={"configurable": {"thread_id": "t3"}},
    )

    assert result["messages"][-1].content == "Sorry, that failed both times."
    assert any(
        isinstance(m, HumanMessage) and m.content == GIVE_UP_PROMPT for m in result["messages"]
    )


async def test_step_budget_gets_a_graceful_final_turn_instead_of_hard_end():
    """Regression test: hitting max_react_steps used to hard-END the graph
    right after an agent message that only requested a tool call, which
    could surface as an empty/stale reply to the caller. It should instead
    get one grace turn (STEP_BUDGET_PROMPT) and end on real summary text,
    without ever executing the tool call that triggered the budget."""
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "good_tool", "args": {"x": "1"}, "id": "call_1"}],
            ),
            AIMessage(content="Budget hit -- here's what I found so far."),
        ]
    )
    graph = build_graph(model, [good_tool], InMemorySaver(), InMemoryStore())

    result = await graph.ainvoke(
        _initial_state(max_react_steps=1),
        config={"configurable": {"thread_id": "t4"}},
    )

    assert result["messages"][-1].content == "Budget hit -- here's what I found so far."
    assert result["budget_stop_issued"] is True
    assert any(
        isinstance(m, HumanMessage) and m.content == STEP_BUDGET_PROMPT for m in result["messages"]
    )
    # The tool call that triggered the budget was never actually executed.
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])


async def test_step_budget_ignores_tool_calls_from_the_grace_turn_itself():
    """If the grace turn disobeys STEP_BUDGET_PROMPT and asks for another
    tool call anyway, the graph must still end rather than granting a
    second grace turn (or worse, looping)."""
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "good_tool", "args": {"x": "1"}, "id": "call_1"}],
            ),
            AIMessage(
                content="one more search, I promise",
                tool_calls=[{"name": "good_tool", "args": {"x": "2"}, "id": "call_2"}],
            ),
        ]
    )
    graph = build_graph(model, [good_tool], InMemorySaver(), InMemoryStore())

    result = await graph.ainvoke(
        _initial_state(max_react_steps=1),
        config={"configurable": {"thread_id": "t5"}},
    )

    assert model.calls == 2
    assert result["messages"][-1].content == "one more search, I promise"
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])
