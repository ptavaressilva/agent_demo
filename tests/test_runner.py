"""Tests for `agent_demo.runner._extract_reply`, the pure function that
picks the caller-facing reply out of the graph's final message list. No
Mongo/Postgres/MCP/Anthropic needed -- this is exactly the part of `run()`
that doesn't depend on them.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from agent_demo.runner import _NO_REPLY_FALLBACK, InvokeRequest, _build_graph_input, _extract_reply


def test_extract_reply_returns_last_text_message():
    messages = [
        HumanMessage(content="find me a house"),
        AIMessage(content="Here's what I found: listing A, listing B."),
    ]
    assert _extract_reply(messages) == "Here's what I found: listing A, listing B."


def test_extract_reply_skips_trailing_tool_call_only_message():
    messages = [
        AIMessage(content="Earlier summary."),
        ToolMessage(content="ok", tool_call_id="1", name="good_tool"),
        AIMessage(content="", tool_calls=[{"name": "good_tool", "args": {}, "id": "2"}]),
    ]
    assert _extract_reply(messages) == "Earlier summary."


def test_extract_reply_falls_back_when_no_ai_text_exists_at_all():
    messages = [
        HumanMessage(content="find me a house"),
        AIMessage(content="", tool_calls=[{"name": "good_tool", "args": {}, "id": "1"}]),
    ]
    assert _extract_reply(messages) == _NO_REPLY_FALLBACK


def test_build_graph_input_returns_fresh_state_for_a_new_turn():
    request = InvokeRequest(
        message="find me a house",
        buyer_id="b1",
        buyer_profile="Family of three, budget EUR 450k",
    )

    graph_input = _build_graph_input(request, session_id="s1")

    assert graph_input["messages"] == [HumanMessage(content="find me a house")]
    assert graph_input["buyer_id"] == "b1"
    assert graph_input["react_steps"] == 0
    assert graph_input["budget_stop_issued"] is False


def test_build_graph_input_returns_a_bare_resume_command_without_reseeding_state():
    request = InvokeRequest(
        buyer_id="b1",
        session_id="s1",
        resume_decision={"action": "approve"},
    )

    graph_input = _build_graph_input(request, session_id="s1")

    assert graph_input == Command(resume={"action": "approve"})


def test_resume_decision_requires_an_existing_session_id():
    with pytest.raises(ValueError):
        InvokeRequest(buyer_id="b1", resume_decision={"action": "approve"})


def test_message_and_buyer_profile_required_when_not_resuming():
    with pytest.raises(ValueError):
        InvokeRequest(buyer_id="b1")
