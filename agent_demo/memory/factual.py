"""Factual memory: a verified-facts knowledge base, separate from candidate
preferences (long-term memory) and conversation history (short-term memory).

Holds things the agent has confirmed while researching -- "Acme Corp's
engineering org is fully remote", "Acme's last posted senior-IC band tops out
at $210k" -- so future sessions don't re-derive them from scratch. Exposed to
the agent as `remember_fact` / `recall_facts` tools (see
`agent_demo.tools.memory_tools`), not through the LangGraph `Store`
injection used for long-term preferences, since facts are freeform research
notes rather than structured per-candidate keys.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient, TEXT
from pymongo.collection import Collection

from agent_demo.config import settings


class FactualMemory:
    def __init__(self, client: MongoClient, collection_name: str = "factual_memory") -> None:
        self._collection: Collection = client[settings.mongo_db_name][collection_name]
        self._collection.create_index("topic")
        self._collection.create_index([("statement", TEXT), ("topic", TEXT)])

    def remember(self, topic: str, statement: str, source_url: str | None = None) -> str:
        """Upsert a fact under `topic`. Returns the fact id."""
        now = datetime.now(timezone.utc)
        doc = {
            "topic": topic,
            "statement": statement,
            "source_url": source_url,
            "updated_at": now,
        }
        result = self._collection.update_one(
            {"topic": topic, "statement": statement},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        fact_id = result.upserted_id or self._collection.find_one(
            {"topic": topic, "statement": statement}, {"_id": 1}
        )["_id"]
        return str(fact_id)

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Free-text search over remembered facts."""
        cursor = (
            self._collection.find({"$text": {"$search": query}}, {"score": {"$meta": "textScore"}})
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )
        return [
            {
                "topic": doc["topic"],
                "statement": doc["statement"],
                "source_url": doc.get("source_url"),
            }
            for doc in cursor
        ]

    def recall_by_topic(self, topic: str, limit: int = 20) -> list[dict[str, Any]]:
        cursor = self._collection.find({"topic": topic}).limit(limit)
        return [
            {
                "topic": doc["topic"],
                "statement": doc["statement"],
                "source_url": doc.get("source_url"),
            }
            for doc in cursor
        ]
