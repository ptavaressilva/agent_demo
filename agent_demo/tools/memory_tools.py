"""Tools exposing factual memory (research notes) and long-term memory
(candidate preferences) to the agent. Short-term memory needs no tools --
it's the graph's own message history, handled transparently by the
checkpointer.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from langgraph.store.base import BaseStore

from agent_demo.memory.factual import FactualMemory


def build_memory_tools(
    factual: FactualMemory, long_term: BaseStore, candidate_id: str
) -> list[BaseTool]:
    namespace = ("candidate", candidate_id, "preferences")

    @tool
    async def remember_fact(topic: str, statement: str, source_url: str = "") -> str:
        """Save a verified fact you learned while researching (e.g. about a
        company, role, or the job market) so future sessions don't have to
        re-research it. `topic` should be a short stable label (e.g. a
        company name) you'd reuse to recall this later.
        """
        fact_id = factual.remember(topic, statement, source_url or None)
        return f"Remembered fact {fact_id} under topic {topic!r}."

    @tool
    async def recall_facts(query: str) -> str:
        """Search previously remembered facts by free-text query. Use this
        before researching a company/topic you might already have notes on.
        """
        facts = factual.recall(query)
        if not facts:
            return "No remembered facts match that query."
        return "\n".join(f"- [{f['topic']}] {f['statement']}" for f in facts)

    @tool
    async def save_candidate_preference(key: str, value: str) -> str:
        """Persist a durable candidate preference learned in conversation
        (e.g. key='min_salary', value='150000', or key='disliked_industry',
        value='crypto') so it carries over to future job-search sessions
        with this candidate.
        """
        await long_term.aput(namespace, key, {"value": value})
        return f"Saved preference {key}={value!r} for future sessions."

    @tool
    async def recall_candidate_preferences() -> str:
        """List all durable preferences saved for this candidate in past
        sessions. Call this early so you don't re-ask for things the
        candidate already told a previous session.
        """
        items = await long_term.asearch(namespace)
        if not items:
            return "No saved preferences for this candidate yet."
        return "\n".join(f"- {item.key}: {item.value.get('value')}" for item in items)

    return [remember_fact, recall_facts, save_candidate_preference, recall_candidate_preferences]
