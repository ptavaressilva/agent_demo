"""Tests for `agent_demo.platform.plan_execute_graph`: the Plan-and-Execute
alternative to `build_react_graph`'s plain ReAct loop. Mirrors
`test_graph_factory.py`'s style (scripted fake model, in-memory
checkpointer/store) but exercises the planner/execute-step/replan/finish
shape instead, including the two guarantees this factory deliberately does
*not* carry over from the ReAct factory: no self-correction critic, and no
grace turn on hitting the step budget (see the module docstring).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agent_demo.platform.plan_execute_graph import (
    NO_PLAN_FALLBACK,
    build_plan_execute_graph,
)

_STATIC_SYSTEM_PROMPT = "You are a test agent."


def _system_prompt_fn(_state: dict) -> str:
    return _STATIC_SYSTEM_PROMPT


class FakeModel:
    """Returns each of `responses` in order, one per `ainvoke` call; the last
    response repeats if the graph calls more times than scripted."""

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


def _initial_state(*, max_react_steps: int = 12) -> dict:
    return {
        "messages": [HumanMessage(content="Find the answer to everything.")],
        "session_id": "s1",
        "max_react_steps": max_react_steps,
        "max_self_correction_retries": 0,
        "react_steps": 0,
        "correction_retries": 0,
        "budget_stop_issued": False,
    }


async def test_single_step_plan_executes_a_tool_then_finishes():
    model = FakeModel(
        [
            AIMessage(content="1. Look something up"),
            AIMessage(
                content="",
                tool_calls=[{"name": "good_tool", "args": {"x": "1"}, "id": "call_1"}],
            ),
            AIMessage(content="Found it."),
            AIMessage(content="FINAL: All done, found the answer."),
        ]
    )
    graph = build_plan_execute_graph(
        model, [good_tool], InMemorySaver(), InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )

    result = await graph.ainvoke(
        _initial_state(), config={"configurable": {"thread_id": "t1"}}
    )

    assert result["messages"][-1].content == "All done, found the answer."
    assert result["plan"] == []
    assert result["past_steps"] == [("Look something up", "Found it.")]
    assert result["react_steps"] == 2


async def test_multi_step_plan_replans_between_steps_then_finishes():
    model = FakeModel(
        [
            AIMessage(content="1. Step one\n2. Step two"),
            AIMessage(content="Did step one."),
            AIMessage(content="1. Step two"),
            AIMessage(content="Did step two."),
            AIMessage(content="FINAL: Combined result."),
        ]
    )
    graph = build_plan_execute_graph(
        model, [good_tool], InMemorySaver(), InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )

    result = await graph.ainvoke(
        _initial_state(), config={"configurable": {"thread_id": "t2"}}
    )

    assert result["messages"][-1].content == "Combined result."
    assert result["past_steps"] == [
        ("Step one", "Did step one."),
        ("Step two", "Did step two."),
    ]


async def test_empty_plan_from_planner_falls_back_without_executing_a_step():
    model = FakeModel([AIMessage(content="")])
    graph = build_plan_execute_graph(
        model, [good_tool], InMemorySaver(), InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )

    result = await graph.ainvoke(
        _initial_state(), config={"configurable": {"thread_id": "t3"}}
    )

    assert result["messages"][-1].content == NO_PLAN_FALLBACK
    assert result["past_steps"] == []
    assert model.calls == 1


async def test_step_budget_ends_the_run_without_a_grace_turn():
    """Unlike build_react_graph's budget_stop node, this factory gives no
    extra model turn to summarize on hitting the budget -- it just stops,
    which is the tradeoff documented in the module docstring."""
    model = FakeModel(
        [
            AIMessage(content="1. Step one\n2. Step two"),
            AIMessage(
                content="",
                tool_calls=[{"name": "good_tool", "args": {"x": "1"}, "id": "call_1"}],
            ),
        ]
    )
    graph = build_plan_execute_graph(
        model, [good_tool], InMemorySaver(), InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )

    result = await graph.ainvoke(
        _initial_state(max_react_steps=1), config={"configurable": {"thread_id": "t4"}}
    )

    # The budget-exhausting turn's own (tool-call-only, textless) message
    # still gets recorded as "Step one"'s result -- so finish reports partial
    # progress rather than claiming nothing happened at all.
    assert "Ran out of step budget" in result["messages"][-1].content
    assert "Step one" in result["messages"][-1].content
    # Only the planner and the one budget-exhausting agent turn ran -- no
    # tool execution and no replan model call.
    assert model.calls == 2


async def test_new_turn_on_same_session_resets_past_steps_and_uses_the_new_objective():
    """Regression test: a session's checkpointed `messages` accumulate across
    turns (the harness appends each new turn's HumanMessage via
    `add_messages`), so a naive `messages[0]`-as-objective read would still
    see turn 1's request on turn 2, and a naive `operator.add`-reduced
    `past_steps` would carry turn 1's completed steps into turn 2's plan.
    Both must reset per turn."""
    model = FakeModel(
        [
            AIMessage(content="1. Turn one step"),
            AIMessage(content="Did turn one step."),
            AIMessage(content="FINAL: Turn one done."),
        ]
    )
    graph = build_plan_execute_graph(
        model, [good_tool], InMemorySaver(), InMemoryStore(), system_prompt_fn=_system_prompt_fn
    )
    config = {"configurable": {"thread_id": "t6"}}

    first = await graph.ainvoke(_initial_state(), config=config)
    assert first["messages"][-1].content == "Turn one done."
    assert first["past_steps"] == [("Turn one step", "Did turn one step.")]

    model._responses = [
        AIMessage(content="1. Turn two step"),
        AIMessage(content="Did turn two step."),
        AIMessage(content="FINAL: Turn two done."),
    ]
    model.calls = 0

    second = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="A brand new request.")],
            "session_id": "s1",
            "max_react_steps": 12,
            "max_self_correction_retries": 0,
            "react_steps": 0,
            "correction_retries": 0,
            "budget_stop_issued": False,
        },
        config=config,
    )

    assert second["messages"][-1].content == "Turn two done."
    # Only turn two's step is recorded -- turn one's did not carry over.
    assert second["past_steps"] == [("Turn two step", "Did turn two step.")]
    assert second["objective"] == "A brand new request."
