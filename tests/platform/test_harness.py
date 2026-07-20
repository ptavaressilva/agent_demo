"""Tests for `agent_demo.platform.harness`: the platform's single enforcement
point, exercised against a trivial `NullAgent` test double that implements
`AgentSpec` and does nothing else on purpose. If the kill switch, tracing,
and budget-ceiling guarantees hold for an agent that supplies no business
logic at all, they hold structurally -- not because any particular agent's
code happened to cooperate.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

import agent_demo.platform.harness as harness_module
from agent_demo.platform.envelope import BaseInvokeEnvelope
from agent_demo.platform.harness import _NO_REPLY_FALLBACK, _build_graph_input, _extract_reply
from agent_demo.platform.kill_switch import AgentKilledError, KillSwitch
from agent_demo.platform.state import BaseAgentState
from tests.platform.test_kill_switch import FakeMongoClient


class _NullRequest(BaseInvokeEnvelope):
    pass


class NullAgent:
    """Does nothing on purpose: no tools, a static prompt, no domain state.
    Proves the platform's guarantees don't depend on an agent cooperating."""

    agent_id = "null-test-agent"
    request_schema = _NullRequest
    state_schema = BaseAgentState
    primary_model = "test-primary"
    fallback_model = "test-fallback"
    default_max_react_steps = 1
    default_max_self_correction_retries = 0

    def build_initial_domain_state(self, request: _NullRequest) -> dict:
        return {}

    def render_system_prompt(self, state: BaseAgentState) -> str:
        return "Always reply 'ok'."

    async def build_tools(self, resources) -> list:
        return []


class _FakeGraph:
    def __init__(self) -> None:
        self.ainvoke_calls: list[dict] = []

    async def ainvoke(self, graph_input, config):
        self.ainvoke_calls.append({"input": graph_input, "config": config})
        return {"messages": [AIMessage(content="ok")]}


async def _fake_build_model(tools, *, primary_model, fallback_model):
    return "fake-model"


def _patch_happy_path(monkeypatch, fake_client, fake_graph, tracing_calls):
    monkeypatch.setattr(harness_module, "get_mongo_client", lambda: fake_client)
    monkeypatch.setattr(
        harness_module, "configure_tracing", lambda project_name: tracing_calls.append(project_name)
    )
    monkeypatch.setattr(harness_module, "build_checkpointer", lambda client: "fake-checkpointer")
    monkeypatch.setattr(harness_module, "MongoLongTermStore", lambda client: object())
    monkeypatch.setattr(harness_module, "build_chat_model_with_fallback", _fake_build_model)
    monkeypatch.setattr(harness_module, "build_react_graph", lambda *a, **k: fake_graph)


# --- Pure function tests (no Mongo/graph needed) ---


def test_extract_reply_returns_last_text_message():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="Here's the answer."),
    ]
    assert _extract_reply(messages) == "Here's the answer."


def test_extract_reply_skips_trailing_tool_call_only_message():
    messages = [
        AIMessage(content="Earlier summary."),
        ToolMessage(content="ok", tool_call_id="1", name="good_tool"),
        AIMessage(content="", tool_calls=[{"name": "good_tool", "args": {}, "id": "2"}]),
    ]
    assert _extract_reply(messages) == "Earlier summary."


def test_extract_reply_falls_back_when_no_ai_text_exists_at_all():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"name": "good_tool", "args": {}, "id": "1"}]),
    ]
    assert _extract_reply(messages) == _NO_REPLY_FALLBACK


def test_build_graph_input_returns_fresh_state_for_a_new_turn():
    request = _NullRequest(message="hello")

    graph_input = _build_graph_input(
        NullAgent(), request, session_id="s1", max_react_steps=12, max_self_correction_retries=2
    )

    assert graph_input["messages"] == [HumanMessage(content="hello")]
    assert graph_input["session_id"] == "s1"
    assert graph_input["max_react_steps"] == 12
    assert graph_input["react_steps"] == 0
    assert graph_input["budget_stop_issued"] is False


def test_build_graph_input_returns_a_bare_resume_command_without_reseeding_state():
    request = _NullRequest(session_id="s1", resume_decision={"action": "approve"})

    graph_input = _build_graph_input(
        NullAgent(), request, session_id="s1", max_react_steps=12, max_self_correction_retries=2
    )

    assert graph_input == Command(resume={"action": "approve"})


def test_resume_decision_requires_an_existing_session_id():
    with pytest.raises(ValueError):
        _NullRequest(resume_decision={"action": "approve"})


def test_message_required_when_not_resuming():
    with pytest.raises(ValueError):
        _NullRequest()


# --- harness.run() enforcement tests ---


async def test_run_raises_agent_killed_error_before_doing_any_other_work(monkeypatch):
    """The kill-switch check must happen before any resource/tool setup --
    proven here by making every step after it explode if reached."""
    fake_client = FakeMongoClient()
    KillSwitch(fake_client).set_killed(True, reason="on fire", actor="pedro")

    monkeypatch.setattr(harness_module, "get_mongo_client", lambda: fake_client)
    monkeypatch.setattr(harness_module, "configure_tracing", lambda project_name: None)

    def _boom(*args, **kwargs):
        raise AssertionError("build_checkpointer should not run once the kill switch is engaged")

    monkeypatch.setattr(harness_module, "build_checkpointer", _boom)

    with pytest.raises(AgentKilledError) as exc_info:
        await harness_module.run(NullAgent(), {"message": "hi"})

    assert exc_info.value.reason == "on fire"


async def test_run_kill_switch_also_blocks_resumes(monkeypatch):
    fake_client = FakeMongoClient()
    KillSwitch(fake_client).set_killed(True, reason="on fire", actor="pedro")

    monkeypatch.setattr(harness_module, "get_mongo_client", lambda: fake_client)
    monkeypatch.setattr(harness_module, "configure_tracing", lambda project_name: None)
    monkeypatch.setattr(
        harness_module,
        "build_checkpointer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    with pytest.raises(AgentKilledError):
        await harness_module.run(
            NullAgent(), {"session_id": "s1", "resume_decision": {"action": "approve"}}
        )


async def test_configure_tracing_runs_exactly_once_per_run_with_the_agent_id(monkeypatch):
    fake_client = FakeMongoClient()
    tracing_calls: list[str] = []
    fake_graph = _FakeGraph()
    _patch_happy_path(monkeypatch, fake_client, fake_graph, tracing_calls)

    await harness_module.run(NullAgent(), {"message": "hi"})

    assert tracing_calls == ["null-test-agent"]


async def test_oversized_max_react_steps_request_clamps_to_the_platform_ceiling(monkeypatch):
    fake_client = FakeMongoClient()
    tracing_calls: list[str] = []
    fake_graph = _FakeGraph()
    _patch_happy_path(monkeypatch, fake_client, fake_graph, tracing_calls)
    monkeypatch.setattr(harness_module.platform_settings, "max_react_steps_ceiling", 5)

    await harness_module.run(NullAgent(), {"message": "hi", "max_react_steps": 50})

    captured_input = fake_graph.ainvoke_calls[0]["input"]
    assert captured_input["max_react_steps"] == 5


async def test_recursion_limit_is_platform_derived_regardless_of_agent_defaults(monkeypatch):
    """NullAgent's own default_max_react_steps is 1 -- the recursion_limit
    passed to graph.ainvoke must come from the platform ceiling/multiplier,
    not from anything the agent supplied."""
    fake_client = FakeMongoClient()
    tracing_calls: list[str] = []
    fake_graph = _FakeGraph()
    _patch_happy_path(monkeypatch, fake_client, fake_graph, tracing_calls)
    monkeypatch.setattr(harness_module.platform_settings, "max_react_steps_ceiling", 5)
    monkeypatch.setattr(harness_module.platform_settings, "recursion_limit_multiplier", 4)

    await harness_module.run(NullAgent(), {"message": "hi"})

    captured_config = fake_graph.ainvoke_calls[0]["config"]
    assert captured_config["recursion_limit"] == 20


async def test_run_returns_pending_approval_when_graph_interrupts(monkeypatch):
    fake_client = FakeMongoClient()
    tracing_calls: list[str] = []

    class _InterruptingGraph(_FakeGraph):
        async def ainvoke(self, graph_input, config):
            await super().ainvoke(graph_input, config)
            return {
                "messages": [HumanMessage(content="hi")],
                "__interrupt__": [type("I", (), {"value": {"action": "do_thing"}})()],
            }

    fake_graph = _InterruptingGraph()
    _patch_happy_path(monkeypatch, fake_client, fake_graph, tracing_calls)

    result = await harness_module.run(NullAgent(), {"message": "hi"})

    assert result.pending_approval == {"action": "do_thing"}
