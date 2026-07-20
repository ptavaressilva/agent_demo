"""A single trivial tool, backed by an in-memory dict -- no Mongo/Postgres/MCP
dependency at all, to keep this agent's footprint minimal."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

_FAQ_ANSWERS = {
    "kill switch": (
        "An operator can pause any registered agent instantly via the "
        "Mongo-backed kill switch, checked at the start of every invocation "
        "regardless of which agent is running."
    ),
    "tracing": (
        "Every agent's LLM/tool/graph calls are traced to Arize AX "
        "automatically -- configure_tracing() runs before any agent code, "
        "with no per-agent opt-in required."
    ),
    "gateway": (
        "All model calls are routed through the LiteLLM gateway, never "
        "directly to a provider -- an agent picks model names, not a "
        "client."
    ),
}


def build_faq_tools() -> list[BaseTool]:
    @tool
    async def answer_faq(topic: str) -> str:
        """Look up the on-file answer for a known FAQ topic (e.g. 'kill
        switch', 'tracing', 'gateway'). Returns an error string if the topic
        isn't on file -- don't make up an answer in that case."""
        answer = _FAQ_ANSWERS.get(topic.strip().lower())
        if answer is None:
            return f"Error: no FAQ answer on file for topic {topic!r}."
        return answer

    return [answer_faq]
