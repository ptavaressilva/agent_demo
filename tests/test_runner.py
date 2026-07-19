"""Tests for `agent_demo.runner._extract_reply`, the pure function that
picks the caller-facing reply out of the graph's final message list. No
Mongo/Postgres/MCP/Anthropic needed -- this is exactly the part of `run()`
that doesn't depend on them.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_demo.runner import _NO_REPLY_FALLBACK, _extract_reply


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
