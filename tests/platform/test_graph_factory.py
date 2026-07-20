"""Tests for the platform's shared ReAct+critic graph factory: the plain
tool-use loop, the self-correction critic (retry, then give up), and the
step-budget grace turn. All run against a scripted fake model and LangGraph's
in-memory checkpointer/store -- no Mongo/Postgres/MCP/gateway needed, which is
exactly the point of keeping `agent_demo/platform/graph_factory.py` free of
any direct dependency on those, and free of any notion of what a specific
agent's domain state looks like (hence no buyer_id/buyer_profile-style fields
in this test's state dicts -- those belong to a specific agent's own state
schema, see tests/agents/house_search/).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt

from agent_demo.platform.graph_factory import (
    CORRECTION_PROMPT,
    GIVE_UP_PROMPT,
    STEP_BUDGET_PROMPT,
    build_react_graph,
)

_STATIC_SYSTEM_PROMPT = "You are a test agent."


def _system_prompt_fn(_state: dict) -> str:
    return _STATIC_SYSTEM_PROMPT


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


@tool
async def approval_tool(x: str) -> str:
    """A tool that pauses for human approval before acting, mirroring
    draft_viewing_request's interrupt()-before-side-effect shape."""
    decision = interrupt({"action": "approval_tool", "x": x})
    if not isinstance(decision, dict) or decision.get("action") != "approve":
        return "declined"
    return f"approved:{decision.get('x', x)}"


def _initial_state(
    *, max_react_steps: int = 12, max_self_correction_retries: int = 2
) -> dict:
    return {
        "messages": [HumanMessage(content="hello")],
        "session_id": "s1",
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
            AIMessage(content="All done."),
        ]
    )
    graph = build_react_graph(
        model, [good_tool], InMemorySaver(), InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )

    result = await graph.ainvoke(
        _initial_state(), config={"configurable": {"thread_id": "t1"}}
    )

    assert result["messages"][-1].content == "All done."
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
            AIMessage(content="Fixed it."),
        ]
    )
    graph = build_react_graph(
        model, [bad_tool, good_tool], InMemorySaver(), InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )

    result = await graph.ainvoke(
        _initial_state(), config={"configurable": {"thread_id": "t2"}}
    )

    assert result["messages"][-1].content == "Fixed it."
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
    graph = build_react_graph(
        model, [bad_tool], InMemorySaver(), InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )

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
    graph = build_react_graph(
        model, [good_tool], InMemorySaver(), InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )

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
    graph = build_react_graph(
        model, [good_tool], InMemorySaver(), InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )

    result = await graph.ainvoke(
        _initial_state(max_react_steps=1),
        config={"configurable": {"thread_id": "t5"}},
    )

    assert model.calls == 2
    assert result["messages"][-1].content == "one more search, I promise"
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])


async def test_tool_interrupt_pauses_until_resumed_then_completes_on_approve():
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "approval_tool", "args": {"x": "1"}, "id": "call_1"}],
            ),
            AIMessage(content="Done after approval."),
        ]
    )
    graph = build_react_graph(
        model, [approval_tool], InMemorySaver(), InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )
    config = {"configurable": {"thread_id": "t6"}}

    paused = await graph.ainvoke(_initial_state(), config=config)

    assert paused["__interrupt__"][0].value == {"action": "approval_tool", "x": "1"}
    # The graph genuinely stopped before running the tool and before the
    # agent's second scripted turn -- not just before returning.
    assert model.calls == 1

    resumed = await graph.ainvoke(Command(resume={"action": "approve"}), config=config)

    assert resumed["messages"][-1].content == "Done after approval."
    assert any(
        isinstance(m, ToolMessage) and m.content == "approved:1" for m in resumed["messages"]
    )


async def test_tool_interrupt_resumes_on_a_freshly_built_graph_object():
    """Production resumes across separate AgentCore invocations, each
    building a brand-new `graph` object from a shared (Mongo-backed)
    checkpointer -- never the same in-process `graph` twice. Prove resume
    depends only on the checkpointer, not on any in-process graph state, by
    building two independent `build_react_graph` objects here that share
    nothing but the checkpointer instance."""
    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "t8"}}

    model_1 = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "approval_tool", "args": {"x": "1"}, "id": "call_1"}],
            )
        ]
    )
    graph_1 = build_react_graph(
        model_1, [approval_tool], checkpointer, InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )
    paused = await graph_1.ainvoke(_initial_state(), config=config)
    assert paused["__interrupt__"][0].value == {"action": "approval_tool", "x": "1"}

    model_2 = FakeModel([AIMessage(content="Done after approval.")])
    graph_2 = build_react_graph(
        model_2, [approval_tool], checkpointer, InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )
    resumed = await graph_2.ainvoke(Command(resume={"action": "approve"}), config=config)

    assert resumed["messages"][-1].content == "Done after approval."
    assert any(
        isinstance(m, ToolMessage) and m.content == "approved:1" for m in resumed["messages"]
    )


async def test_tool_interrupt_reject_skips_the_side_effect():
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "approval_tool", "args": {"x": "1"}, "id": "call_1"}],
            ),
            AIMessage(content="Okay, skipped that one."),
        ]
    )
    graph = build_react_graph(
        model, [approval_tool], InMemorySaver(), InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )
    config = {"configurable": {"thread_id": "t7"}}

    await graph.ainvoke(_initial_state(), config=config)
    resumed = await graph.ainvoke(Command(resume={"action": "reject"}), config=config)

    assert resumed["messages"][-1].content == "Okay, skipped that one."
    assert any(isinstance(m, ToolMessage) and m.content == "declined" for m in resumed["messages"])
