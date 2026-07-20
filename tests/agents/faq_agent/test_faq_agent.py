"""Tests for the FAQ agent -- the minimal second agent. Mostly a smoke test
that its `AgentSpec` implementation is self-contained: no Mongo/Postgres/MCP
dependency is needed to build its tools or render its state, unlike
house-search's.
"""

from __future__ import annotations

from agent_demo.agents.faq_agent.spec import FaqAgent, FaqAgentRequest
from agent_demo.agents.faq_agent.tools import build_faq_tools


async def test_answer_faq_returns_the_on_file_answer():
    tool = next(t for t in build_faq_tools() if t.name == "answer_faq")

    result = await tool.ainvoke({"topic": "Kill Switch"})

    assert "operator" in result.lower()


async def test_answer_faq_reports_an_error_for_an_unknown_topic():
    tool = next(t for t in build_faq_tools() if t.name == "answer_faq")

    result = await tool.ainvoke({"topic": "quantum entanglement"})

    assert result.startswith("Error:")


def test_faq_agent_builds_tools_with_no_shared_resources():
    agent = FaqAgent()

    tools = _run(agent.build_tools(resources=None))

    assert [t.name for t in tools] == ["answer_faq"]


def test_faq_agent_request_schema_accepts_a_bare_message():
    request = FaqAgentRequest(message="What does the kill switch do?")

    assert request.topic is None


def _run(coro):
    import asyncio

    return asyncio.run(coro)
