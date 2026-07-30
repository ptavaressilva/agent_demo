"""An alternative loop pattern to `build_react_graph`'s plain ReAct loop:
Plan-and-Execute. Where ReAct interleaves "think a little, act a little" on
every turn, this factory front-loads a `planner` node that lays out the full
list of steps once, then executes each step (itself a small agent<->tools
loop, one step at a time) before a `replan` node decides whether to revise
the remaining steps or produce the final answer.

This exists to prove the escape hatch documented in `platform.spec.AgentSpec`
is real: an `AgentSpec` sets `graph_factory = build_plan_execute_graph`
instead of leaving it unset, and `harness.run` builds this graph instead of
the default ReAct one -- with identical kill switch/tracing/gateway
routing/budget-ceiling/`recursion_limit` enforcement around it either way,
since those live in `harness.run`, not in whichever factory it calls.

    START -> planner -> agent -> [tools -> agent]* -> record_step -> replan
                                                          -> agent (next step)
                                                          -> finish -> END

Two things this factory deliberately does *not* carry over from
`build_react_graph`, illustrating the tradeoff of picking a different
factory: there is no self-correction critic (a failing tool call just
becomes part of that step's recorded result, for the model to notice on the
next replan), and hitting the step budget mid-step ends the run directly
through `finish` rather than granting one more model turn to summarize first.
An agent that wants those guarantees back has to build its own equivalents.

`max_react_steps` is reused here as a whole-run step budget (incremented
once per agent turn, across every plan step) rather than introducing a
separate ceiling -- the platform's `clamp_budget`/`recursion_limit` backstops
already reason about that field, and adding a second one would let an agent
quietly escape the platform ceiling just by picking a different factory.
"""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore

from agent_demo.platform.state import BaseAgentState


class PlanExecuteState(BaseAgentState):
    """Extra state this factory needs on top of `BaseAgentState`. An agent
    using `build_plan_execute_graph` must pass this (or a further extension
    of it) as its `state_schema`, the same way an agent using the ReAct
    factory extends `BaseAgentState` directly.

    None of these fields use a LangGraph reducer (e.g. `operator.add`) --
    every write below is a full replacement value, including `past_steps`
    (nodes append by writing `state["past_steps"] + [...]`, not by returning
    just the new tail). A reducer would accumulate across turns as well as
    within one, and `planner_node` resetting `past_steps` to `[]` at the
    start of a new turn would then be absorbed into a no-op instead of
    actually clearing prior turns' completed steps.

    `objective` exists because `state["messages"]` is the *whole* session
    history across every turn (the harness's checkpointer persists it via
    `add_messages`), so `messages[0]` is only that turn's request on the
    very first turn -- `planner_node` captures the current turn's request
    once, into this field, before it starts appending its own step-prompt
    messages that would otherwise make "the objective" ambiguous to locate.
    """

    plan: list[str]
    past_steps: list[tuple[str, str]]
    response: str | None
    objective: str


PLANNER_PROMPT = """Break the objective below into a short numbered list of \
concrete steps that could be executed one at a time, calling a tool when a \
step needs one. Respond with ONLY the numbered list (one step per line) and \
nothing else.

Objective: {objective}"""

STEP_PROMPT = """Overall objective: {objective}

Steps already completed:
{completed}

Execute this step now: {step}

Call a tool if this step needs one. Once you have what this step needs, \
reply with a short text summary of the result (no further tool calls) so \
the plan can move on to the next step."""

REPLAN_PROMPT = """Overall objective: {objective}

Steps completed so far:
{completed}

If the objective is now fully satisfied, respond with exactly:
FINAL: <the final answer to give the user>

Otherwise, respond with ONLY an updated numbered list of the remaining \
steps -- drop anything already done, and add new ones if the completed \
steps revealed the original plan was incomplete."""

NO_PLAN_FALLBACK = "(Could not produce a plan or a final answer for this request.)"

_STEP_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*")


def _parse_plan(text: str) -> list[str]:
    steps = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        steps.append(_STEP_PREFIX_RE.sub("", line).strip())
    return steps


def build_plan_execute_graph(
    model: Runnable,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    system_prompt_fn: Callable[[BaseAgentState], str],
    state_schema: type = PlanExecuteState,
) -> CompiledStateGraph:
    """Same call convention as `graph_factory.build_react_graph` -- see that
    function's docstring for why node functions take `state: dict` rather
    than `state: PlanExecuteState`."""

    def _completed_summary(state: dict) -> str:
        if not state["past_steps"]:
            return "(none yet)"
        return "\n".join(f"- {step}: {result}" for step, result in state["past_steps"])

    async def planner_node(state: dict) -> dict:
        # `state["messages"][-1]` is this turn's request specifically:
        # planner is always the first node to touch `messages` on a given
        # turn, before anything else (step prompts, tool results) gets
        # appended -- see `PlanExecuteState`'s docstring on why this can't
        # just be `messages[0]`.
        objective = state["messages"][-1].content
        system = SystemMessage(content=system_prompt_fn(state))
        request = HumanMessage(content=PLANNER_PROMPT.format(objective=objective))
        response = await model.ainvoke([system, request])
        text = response.content if isinstance(response.content, str) else ""
        return {
            "objective": objective,
            "plan": _parse_plan(text),
            "past_steps": [],
            "response": None,
        }

    def route_after_planner(state: dict) -> str:
        return "agent" if state["plan"] else "finish"

    async def agent_node(state: dict) -> dict:
        prompt = STEP_PROMPT.format(
            objective=state["objective"],
            completed=_completed_summary(state),
            step=state["plan"][0],
        )
        system = SystemMessage(content=system_prompt_fn(state))
        human = HumanMessage(content=prompt)
        response = await model.ainvoke([system, *state["messages"], human])
        return {"messages": [human, response], "react_steps": state["react_steps"] + 1}

    def route_after_agent(state: dict) -> str:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        if wants_tools and state["react_steps"] < state["max_react_steps"]:
            return "tools"
        return "record_step"

    async def record_step_node(state: dict) -> dict:
        last = state["messages"][-1]
        result = last.content if isinstance(last.content, str) else str(last.content)
        step = state["plan"][0]
        return {
            "plan": state["plan"][1:],
            "past_steps": state["past_steps"] + [(step, result)],
        }

    async def replan_node(state: dict) -> dict:
        if state["react_steps"] >= state["max_react_steps"]:
            # No grace turn here (unlike build_react_graph's budget_stop) --
            # see the module docstring's tradeoff note. `finish_node` still
            # reports whatever steps did complete, just without a
            # model-generated summary of them.
            return {"plan": []}
        system = SystemMessage(content=system_prompt_fn(state))
        request = HumanMessage(
            content=REPLAN_PROMPT.format(
                objective=state["objective"], completed=_completed_summary(state)
            )
        )
        response = await model.ainvoke([system, request])
        text = response.content if isinstance(response.content, str) else ""
        if text.strip().startswith("FINAL:"):
            return {"plan": [], "response": text.split("FINAL:", 1)[1].strip()}
        return {"plan": _parse_plan(text)}

    def route_after_replan(state: dict) -> str:
        return "agent" if state["plan"] else "finish"

    async def finish_node(state: dict) -> dict:
        if state.get("response"):
            content = state["response"]
        elif state["past_steps"]:
            content = (
                "Ran out of step budget before finishing the plan. "
                "Completed so far:\n" + _completed_summary(state)
            )
        else:
            content = NO_PLAN_FALLBACK
        return {"messages": [AIMessage(content=content)]}

    graph = StateGraph(state_schema)
    graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("record_step", record_step_node)
    graph.add_node("replan", replan_node)
    graph.add_node("finish", finish_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner", route_after_planner, {"agent": "agent", "finish": "finish"}
    )
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "record_step": "record_step"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("record_step", "replan")
    graph.add_conditional_edges(
        "replan", route_after_replan, {"agent": "agent", "finish": "finish"}
    )
    graph.add_edge("finish", END)

    return graph.compile(checkpointer=checkpointer, store=store)
