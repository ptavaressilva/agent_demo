"""Regression test for a real LangGraph gotcha: `build_react_graph`'s node
functions must NOT be annotated with a narrower TypedDict (e.g.
`BaseAgentState`) than the `state_schema` the graph was built with (e.g.
`HouseSearchState`) -- LangGraph treats a node function's own type
annotation as a per-node *input schema* and silently filters the state dict
down to only the fields that narrower schema declares before the node ever
sees it. That would drop every domain field (`buyer_id`, `buyer_profile`)
before `render_system_prompt` gets a chance to read them, and no other test
in this repo runs a real (non-monkeypatched) graph with a subclassed state
schema to catch it -- see agent_demo/platform/graph_factory.py's docstring
on `build_react_graph` for where this is guarded against.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from langchain_core.tools import tool

from agent_demo.agents.house_search.state import HouseSearchState
from agent_demo.platform.graph_factory import build_react_graph


class FakeModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self.calls = 0

    async def ainvoke(self, _messages, *args, **kwargs) -> AIMessage:
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[index]


@tool
async def approval_tool(x: str) -> str:
    """A tool that pauses for approval, to exercise the resume path too."""
    decision = interrupt({"x": x})
    return f"approved:{decision}"


def _initial_state() -> dict:
    return {
        "messages": [HumanMessage(content="hi")],
        "session_id": "s1",
        "max_react_steps": 12,
        "max_self_correction_retries": 2,
        "react_steps": 0,
        "correction_retries": 0,
        "budget_stop_issued": False,
        "buyer_id": "b1",
        "buyer_profile": "Family of three, budget EUR 450k",
    }


async def test_domain_state_fields_reach_system_prompt_fn_on_a_fresh_turn():
    captured_states: list[dict] = []

    def system_prompt_fn(state: dict) -> str:
        captured_states.append(dict(state))
        return f"prompt for {state.get('buyer_profile')}"

    model = FakeModel([AIMessage(content="All done.")])
    graph = build_react_graph(
        model,
        [],
        InMemorySaver(),
        InMemoryStore(),
        system_prompt_fn=system_prompt_fn,
        state_schema=HouseSearchState,
    )

    result = await graph.ainvoke(
        _initial_state(), config={"configurable": {"thread_id": "t-fresh"}}
    )

    assert captured_states[-1]["buyer_profile"] == "Family of three, budget EUR 450k"
    assert captured_states[-1]["buyer_id"] == "b1"
    assert result["buyer_profile"] == "Family of three, budget EUR 450k"


async def test_domain_state_fields_survive_a_checkpoint_resume():
    captured_states: list[dict] = []

    def system_prompt_fn(state: dict) -> str:
        captured_states.append(dict(state))
        return f"prompt for {state.get('buyer_profile')}"

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
        model,
        [approval_tool],
        InMemorySaver(),
        InMemoryStore(),
        system_prompt_fn=system_prompt_fn,
        state_schema=HouseSearchState,
    )
    config = {"configurable": {"thread_id": "t-resume"}}

    await graph.ainvoke(_initial_state(), config=config)
    captured_states.clear()

    result = await graph.ainvoke(Command(resume={"action": "approve"}), config=config)

    assert captured_states[-1]["buyer_profile"] == "Family of three, budget EUR 450k"
    assert result["buyer_profile"] == "Family of three, budget EUR 450k"
