"""Long-term memory: durable, cross-session state about a buyer.

Backed by a `langgraph.store.base.BaseStore` implementation over a plain
Mongo collection. Unlike short-term memory (scoped to one session's
checkpoint), items here are addressed by a hierarchical namespace
(e.g. `("buyer", buyer_id, "preferences")`) and are meant to be read
and written across many separate sessions -- house-search preferences learned
in one conversation should inform the next one.

Only `batch`/`abatch` are abstract on `BaseStore`; everything else (get, put,
search, ...) is implemented in terms of them, so that's all we implement
here. The underlying driver (`pymongo`) is sync; `abatch` wraps `batch` in a
thread, matching the pattern LangGraph's own `MongoDBSaver` uses for its
async methods.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from pymongo import MongoClient
from pymongo.collection import Collection

from agent_demo.config import settings


class MongoLongTermStore(BaseStore):
    """Minimal namespace/key/value store over a Mongo collection.

    Search is filter-only (no vector/semantic ranking) -- adequate for
    structured long-term facts like "preferred neighborhoods" or "budget ceiling".
    """

    def __init__(self, client: MongoClient, collection_name: str = "long_term_memory") -> None:
        self._collection: Collection = client[settings.mongo_db_name][collection_name]
        self._collection.create_index("namespace")
        self._collection.create_index([("namespace", 1), ("key", 1)], unique=True)

    @staticmethod
    def _doc_id(namespace: tuple[str, ...], key: str) -> str:
        return "/".join(namespace) + f"::{key}"

    def _to_item(self, doc: dict[str, Any]) -> Item:
        return Item(
            value=doc["value"],
            key=doc["key"],
            namespace=tuple(doc["namespace"]),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )

    def batch(self, ops: Sequence[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                doc = self._collection.find_one(
                    {"_id": self._doc_id(op.namespace, op.key)}
                )
                results.append(self._to_item(doc) if doc else None)
            elif isinstance(op, PutOp):
                if op.value is None:
                    self._collection.delete_one(
                        {"_id": self._doc_id(op.namespace, op.key)}
                    )
                    results.append(None)
                else:
                    now = datetime.now(timezone.utc)
                    self._collection.update_one(
                        {"_id": self._doc_id(op.namespace, op.key)},
                        {
                            "$set": {
                                "namespace": list(op.namespace),
                                "key": op.key,
                                "value": op.value,
                                "updated_at": now,
                            },
                            "$setOnInsert": {"created_at": now},
                        },
                        upsert=True,
                    )
                    results.append(None)
            elif isinstance(op, SearchOp):
                # Prefix match: namespace array must start with namespace_prefix.
                # NOTE: this pulls every document matching the coarse Mongo
                # `$all` pre-filter into Python before the exact-prefix check
                # and offset/limit slice below -- fine at demo scale (a
                # handful of buyers/preferences), but not a real pagination
                # strategy. A production store should push the prefix match
                # and slicing into the Mongo query itself.
                prefix = list(op.namespace_prefix)
                cursor = self._collection.find(
                    {} if not prefix else {"namespace": {"$all": prefix}}
                )
                items = [
                    self._to_item(doc)
                    for doc in cursor
                    if list(doc["namespace"])[: len(prefix)] == prefix
                ]
                items = items[op.offset : op.offset + op.limit]
                results.append(
                    [
                        SearchItem(
                            value=i.value,
                            key=i.key,
                            namespace=i.namespace,
                            created_at=i.created_at,
                            updated_at=i.updated_at,
                        )
                        for i in items
                    ]
                )
            elif isinstance(op, ListNamespacesOp):
                namespaces = self._collection.distinct("namespace")
                unique = sorted({tuple(ns) for ns in namespaces})
                results.append(unique[op.offset : op.offset + op.limit])
            else:  # pragma: no cover - defensive
                raise TypeError(f"Unsupported store op: {type(op)}")
        return results

    async def abatch(self, ops: Sequence[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, ops)
