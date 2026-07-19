"""Tools exposing factual memory (research notes) and long-term memory
(buyer preferences) to the agent. Short-term memory needs no tools -- it's
the graph's own message history, handled transparently by the checkpointer.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from langgraph.store.base import BaseStore

from agent_demo.memory.factual import FactualMemory


def build_memory_tools(
    factual: FactualMemory, long_term: BaseStore, buyer_id: str
) -> list[BaseTool]:
    namespace = ("buyer", buyer_id, "preferences")

    @tool
    async def remember_fact(topic: str, statement: str, source_url: str = "") -> str:
        """Save a verified fact you learned while researching (e.g. about a
        neighborhood, building, or the housing market) so future sessions
        don't have to re-research it. `topic` should be a short stable label
        (e.g. a neighborhood name) you'd reuse to recall this later.
        """
        fact_id = factual.remember(topic, statement, source_url or None)
        return f"Remembered fact {fact_id} under topic {topic!r}."

    @tool
    async def recall_facts(query: str) -> str:
        """Search previously remembered facts by free-text query. Use this
        before researching a neighborhood/topic you might already have notes on.
        """
        facts = factual.recall(query)
        if not facts:
            return "No remembered facts match that query."
        return "\n".join(f"- [{f['topic']}] {f['statement']}" for f in facts)

    @tool
    async def save_buyer_preference(key: str, value: str) -> str:
        """Persist a durable buyer preference learned in conversation
        (e.g. key='max_budget', value='450000', or key='disliked_neighborhood',
        value='industrial district') so it carries over to future
        house-search sessions with this buyer.
        """
        await long_term.aput(namespace, key, {"value": value})
        return f"Saved preference {key}={value!r} for future sessions."

    @tool
    async def recall_buyer_preferences() -> str:
        """List all durable preferences saved for this buyer in past
        sessions. Call this early so you don't re-ask for things the
        buyer already told a previous session.
        """
        items = await long_term.asearch(namespace)
        if not items:
            return "No saved preferences for this buyer yet."
        return "\n".join(f"- {item.key}: {item.value.get('value')}" for item in items)

    return [remember_fact, recall_facts, save_buyer_preference, recall_buyer_preferences]
