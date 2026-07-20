"""The one place every registered agent's invocation goes through, independent
of how it's hosted (AgentCore, local CLI, tests). This is the part of the
system that's fully testable without deploying anything -- `main.py` is a
thin AgentCore adapter around `run()` below.

`run()` is the platform's enforcement point: it always configures tracing,
always checks the kill switch before any other work, always routes model
calls through the gateway, always clamps budgets to the platform ceiling, and
always builds the graph via the shared ReAct+critic factory -- for whichever
`AgentSpec` it's given. An agent cannot skip any of this because it never
gets a chance to run its own version of this function; it only supplies the
pieces `AgentSpec` asks for (tools, domain state, a system prompt renderer).

Process-wide resources (Mongo client) are created lazily and cached for the
life of the process; only the parts that vary per request (this request's
tools, the graph's own conversation state) are rebuilt per call.
"""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from agent_demo.lib.memory.long_term import MongoLongTermStore
from agent_demo.lib.memory.short_term import build_checkpointer
from agent_demo.platform.budget import clamp_budget
from agent_demo.platform.config import platform_settings
from agent_demo.platform.envelope import BaseInvokeEnvelope, InvokeResult
from agent_demo.platform.graph_factory import build_react_graph
from agent_demo.platform.kill_switch import KillSwitch
from agent_demo.platform.llm import build_chat_model_with_fallback
from agent_demo.platform.mongo import get_mongo_client
from agent_demo.platform.spec import AgentSpec, RequestResources
from agent_demo.platform.tracing import configure_tracing

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

    The factory graph's step-budget/retry-limit grace turns are designed to
    always produce real summary text (see graph_factory.py's `budget_stop`
    node and `GIVE_UP_PROMPT`), but that's a model-compliance guarantee, not
    a structural one -- a grace turn could still reply with only a
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


def _build_graph_input(
    agent: AgentSpec,
    request: BaseInvokeEnvelope,
    session_id: str,
    max_react_steps: int,
    max_self_correction_retries: int,
) -> dict | Command:
    """The input to feed `graph.ainvoke`: a `Command(resume=...)` if this
    call is responding to a pending human-in-the-loop approval, otherwise a
    fresh turn's initial state.

    A resumed run's state comes entirely from the checkpoint LangGraph
    already has for `session_id` -- never merge the two, or you'd be
    re-seeding react_steps/correction_retries/domain state alongside the
    resume value instead of continuing from where the interrupt paused.
    """
    if request.resume_decision is not None:
        return Command(resume=request.resume_decision)
    return {
        "messages": [HumanMessage(content=request.message)],
        "session_id": session_id,
        "max_react_steps": max_react_steps,
        "max_self_correction_retries": max_self_correction_retries,
        "react_steps": 0,
        "correction_retries": 0,
        "budget_stop_issued": False,
        **agent.build_initial_domain_state(request),
    }


async def run(agent: AgentSpec, payload: dict) -> InvokeResult:
    """Handle one agent turn end-to-end for `agent`: build the tool/model/
    graph stack for this request, run the ReAct + self-correction loop, and
    return the agent's reply -- or, if a tool paused for human approval
    (see platform.hitl), return that pending approval instead.

    Raises `AgentKilledError` before doing any other work -- including the
    agent's own resource setup and, notably, before resuming a paused
    approval -- if an operator has engaged the kill switch (see
    agent_demo.platform.kill_switch). This check runs identically for every
    agent; nothing in `AgentSpec` can influence or skip it.
    """
    configure_tracing(project_name=agent.agent_id)

    request = agent.request_schema.model_validate(payload)
    session_id = request.session_id or str(uuid.uuid4())

    mongo_client = get_mongo_client()
    KillSwitch(mongo_client).check()

    checkpointer = build_checkpointer(mongo_client)
    long_term_store = MongoLongTermStore(mongo_client)

    resources = RequestResources(
        mongo_client=mongo_client,
        session_id=session_id,
        request=request,
        long_term_store=long_term_store,
    )
    tools = await agent.build_tools(resources)

    max_react_steps = clamp_budget(
        requested=request.max_react_steps,
        agent_default=agent.default_max_react_steps,
        platform_ceiling=platform_settings.max_react_steps_ceiling,
    )
    max_self_correction_retries = clamp_budget(
        requested=request.max_self_correction_retries,
        agent_default=agent.default_max_self_correction_retries,
        platform_ceiling=platform_settings.max_self_correction_retries_ceiling,
    )

    model = await build_chat_model_with_fallback(
        tools, primary_model=agent.primary_model, fallback_model=agent.fallback_model
    )
    graph = build_react_graph(
        model,
        tools,
        checkpointer,
        long_term_store,
        system_prompt_fn=agent.render_system_prompt,
        state_schema=agent.state_schema,
    )

    graph_config = {
        "configurable": {"thread_id": session_id},
        # Structural backstop, independent of the max_react_steps counter
        # above: no agent code path can raise this. A ReAct+critic turn is
        # several LangGraph super-steps (agent -> tools -> critic), not one,
        # hence the multiplier rather than a 1:1 mapping to step count.
        "recursion_limit": (
            platform_settings.recursion_limit_multiplier * platform_settings.max_react_steps_ceiling
        ),
    }
    graph_input = _build_graph_input(
        agent, request, session_id, max_react_steps, max_self_correction_retries
    )
    result = await graph.ainvoke(graph_input, config=graph_config)

    if pending := result.get("__interrupt__"):
        return InvokeResult(
            session_id=session_id,
            reply=_PENDING_APPROVAL_REPLY,
            message_count=len(result.get("messages", [])),
            pending_approval=pending[0].value,
        )

    reply = _extract_reply(result["messages"])
    return InvokeResult(session_id=session_id, reply=reply, message_count=len(result["messages"]))
