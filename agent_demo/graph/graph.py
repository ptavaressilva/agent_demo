"""The agent's LangGraph: a ReAct loop (agent <-> tools) with an explicit
self-correction step. Plain ReAct would loop agent -> tools -> agent forever
on a bad tool call, re-emitting the same broken request. Here every tool
result passes through `critic_node` first, which detects tool errors and, up
to `settings.max_self_correction_retries` times, injects an explicit
correction instruction before handing control back to the agent -- and
forces a graceful stop once retries are exhausted instead of looping.

    START -> agent -> [tools -> critic -> agent]* -> END
                    (loops while tool calls are made and the step budget allows)
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore

from agent_demo.config import settings
from agent_demo.graph.prompts import CORRECTION_PROMPT, GIVE_UP_PROMPT, SYSTEM_PROMPT
from agent_demo.graph.state import AgentState


def _is_error_tool_message(message: ToolMessage) -> bool:
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


def build_graph(
    model: Runnable,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
) -> CompiledStateGraph:
    async def agent_node(state: AgentState) -> dict:
        system = SystemMessage(
            content=SYSTEM_PROMPT.format(candidate_profile=state["candidate_profile"])
        )
        response = await model.ainvoke([system, *state["messages"]])
        return {"messages": [response], "react_steps": state["react_steps"] + 1}

    def route_after_agent(state: AgentState) -> str:
        if state["react_steps"] >= settings.max_react_steps:
            return END
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    async def critic_node(state: AgentState) -> dict:
        errors = [m for m in _trailing_tool_messages(state["messages"]) if _is_error_tool_message(m)]
        if not errors:
            return {"correction_retries": 0}

        retries_used = state["correction_retries"]
        retries_left = settings.max_self_correction_retries - retries_used
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

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "critic")
    graph.add_edge("critic", "agent")

    return graph.compile(checkpointer=checkpointer, store=store)
