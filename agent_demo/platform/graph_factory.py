"""The standard ReAct loop (agent <-> tools) with an explicit self-correction
step, shared by every agent that uses it. Plain ReAct would loop
agent -> tools -> agent forever on a bad tool call, re-emitting the same
broken request. Here every tool result passes through `critic_node` first,
which detects tool errors and, up to `max_self_correction_retries` times,
injects an explicit correction instruction before handing control back to the
agent -- and forces a graceful stop once retries are exhausted instead of
looping.

Hitting the step budget (`max_react_steps`) gets the same graceful-stop
treatment: rather than hard-ending mid-loop (which can cut off right after
the agent emits a tool-call-only message, before it's said anything to the
caller), one final "wrap it up" turn is injected via `budget_stop_node` so
the agent always gets to produce a real summary before the run ends.

    START -> agent -> [tools -> critic -> agent]* -> END
                    (loops while tool calls are made and the step budget allows)
                    -> budget_stop -> agent -> END   (one grace turn on budget exhaustion)

This graph shape (and the critic/budget-stop guarantee that comes with it) is
what an agent gets by having `harness.run()` build its graph via
`build_react_graph` -- it is not something an agent's own code assembles, so
there is no way for an agent to skip the critic or the graceful-stop grace
turn while still using this factory. An agent needing genuinely non-ReAct
control flow is the one documented escape hatch (not implemented here): it
would forfeit this factory's critic/budget-stop guarantee, though kill
switch/tracing/gateway routing/the `recursion_limit` backstop remain enforced
by `harness.run()` regardless of which graph a request ends up using.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore

from agent_demo.platform.state import BaseAgentState

CORRECTION_PROMPT = """The previous tool call above failed:
{error_summary}

Do not repeat the same call unchanged. Diagnose the cause (e.g. a wrong \
argument, a stale id, a missing prerequisite step) and take a different, \
corrected action. You have {retries_left} correction attempt(s) left for \
this kind of failure before you should give up on that specific step and \
move on."""

GIVE_UP_PROMPT = """You've hit the retry limit for self-correcting the \
failed tool call above. Do not attempt that action again. Summarize what \
you were able to accomplish so far, note what failed and why (briefly), and \
suggest what to do next -- then stop."""

STEP_BUDGET_PROMPT = """You've reached the step budget for this turn -- do \
not call any more tools. Summarize what you've found/done so far, note \
anything you didn't get to, and suggest what to do next -- then stop."""


def _is_error_tool_message(message: ToolMessage) -> bool:
    """Two independent signals, both needed:

    - `status == "error"`: set by `ToolNode` whenever a tool raises
      (including MCP tools -- `langchain_mcp_adapters` raises a
      `ToolException` for `CallToolResult(isError=True)`, which `ToolNode`
      converts to this status).
    - `content.startswith("Error:")`: some hand-written tools (e.g.
      `agents/house_search/tools/postgres_tools.py`) deliberately *return* a
      string instead of raising, so they don't set `status="error"`. Any
      tool that follows the same "return, don't raise" pattern must also
      follow this "Error: ..." convention, or the critic won't see it.
    """
    if getattr(message, "status", "success") == "error":
        return True
    content = message.content if isinstance(message.content, str) else str(message.content)
    return content.strip().startswith("Error:")


def _trailing_tool_messages(messages: list) -> list[ToolMessage]:
    """Tool messages produced by the most recent tool-execution step (i.e.
    everything after the last AIMessage)."""
    trailing: list[ToolMessage] = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            trailing.append(message)
        elif isinstance(message, AIMessage):
            break
    return list(reversed(trailing))


def build_react_graph(
    model: Runnable,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    system_prompt_fn: Callable[[BaseAgentState], str],
    state_schema: type = BaseAgentState,
) -> CompiledStateGraph:
    """`system_prompt_fn` renders the system prompt from the current graph
    state on every agent turn (not just the first) -- so it keeps working
    across a resumed human-in-the-loop approval, where the harness has no
    fresh request to render a prompt from, only whatever domain state the
    agent seeded at the start of the session via `build_initial_domain_state`
    and that the checkpointer has persisted since.

    `state_schema` should be the agent's own state TypedDict (extending
    `BaseAgentState` with whatever domain fields it needs, e.g.
    `HouseSearchState`) -- LangGraph derives its state channels from this
    schema, so domain fields an agent seeds via `build_initial_domain_state`
    must be declared on the schema passed here to actually persist across
    turns/resumes, not just on `BaseAgentState`.

    Node functions below deliberately take `state: dict`, not
    `state: BaseAgentState` -- annotating a node with a narrower TypedDict
    than the graph's own `state_schema` makes LangGraph treat that
    annotation as a *per-node input schema*, silently filtering the state
    dict down to only the fields `BaseAgentState` declares before the node
    ever sees it. That would drop every domain field (e.g. `buyer_profile`)
    before `system_prompt_fn` gets a chance to read it."""

    async def agent_node(state: dict) -> dict:
        system = SystemMessage(content=system_prompt_fn(state))
        response = await model.ainvoke([system, *state["messages"]])
        return {"messages": [response], "react_steps": state["react_steps"] + 1}

    def route_after_agent(state: dict) -> str:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)

        if state["react_steps"] >= state["max_react_steps"]:
            # Give the agent exactly one grace turn to summarize instead of
            # hard-ending mid-loop (e.g. right after a tool-call-only
            # message). `budget_stop_issued` guards against ignoring that
            # grace turn's own tool calls forever if it disobeys the prompt.
            if wants_tools and not state.get("budget_stop_issued", False):
                return "budget_stop"
            return END
        if wants_tools:
            return "tools"
        return END

    async def budget_stop_node(state: dict) -> dict:
        return {
            "messages": [HumanMessage(content=STEP_BUDGET_PROMPT)],
            "budget_stop_issued": True,
        }

    async def critic_node(state: dict) -> dict:
        errors = [m for m in _trailing_tool_messages(state["messages"]) if _is_error_tool_message(m)]
        if not errors:
            return {"correction_retries": 0}

        retries_used = state["correction_retries"]
        retries_left = state["max_self_correction_retries"] - retries_used
        error_summary = "\n".join(f"- {m.name}: {m.content}" for m in errors)

        if retries_left <= 0:
            return {"messages": [HumanMessage(content=GIVE_UP_PROMPT)]}

        correction = CORRECTION_PROMPT.format(
            error_summary=error_summary, retries_left=retries_left - 1
        )
        return {
            "messages": [HumanMessage(content=correction)],
            "correction_retries": retries_used + 1,
        }

    graph = StateGraph(state_schema)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("critic", critic_node)
    graph.add_node("budget_stop", budget_stop_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "budget_stop": "budget_stop", END: END}
    )
    graph.add_edge("tools", "critic")
    graph.add_edge("critic", "agent")
    graph.add_edge("budget_stop", "agent")

    return graph.compile(checkpointer=checkpointer, store=store)
