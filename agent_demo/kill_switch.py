"""Kill switch: an operator-controlled stop for the agent, checked at the
start of every invocation -- including resumes of a paused human-in-the-loop
approval -- so an incident can be contained without a redeploy.

State lives in Mongo (`kill_switch` collection, single `_id="global"`
document) rather than an env var or in-process flag: it's shared correctly
across every running instance and survives restarts, so flipping it during
an incident takes effect for all traffic within `_CACHE_TTL_SECONDS`
instead of only after a redeploy.

`status()` caches its result in-process for `_CACHE_TTL_SECONDS` so a burst
of invocations doesn't turn into a burst of Mongo round trips on every
`run()` call. `set_killed()` invalidates the cache so the operator making
the change sees it take effect immediately; other already-running
processes catch up once their cached entry expires.

Scope: this gates entry to a new turn/resume. It does not interrupt a turn
that's already mid-run -- there's no per-step check inside the ReAct loop.
That's bounded anyway by `max_react_steps`, so a single in-flight run
finishing is an acceptable tradeoff for not adding a check on every graph
step.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection

from agent_demo.config import settings

_DOC_ID = "global"
_CACHE_TTL_SECONDS = 5.0


class AgentKilledError(Exception):
    """Raised by `run()` when the kill switch is engaged. Carries the
    operator-supplied reason so callers/logs can surface why."""

    def __init__(self, reason: str | None) -> None:
        self.reason = reason
        super().__init__(reason or "Agent is currently disabled by the kill switch.")


class KillSwitch:
    # Shared across instances (a new KillSwitch is constructed per `run()`
    # call) but scoped per Mongo client + collection, keyed by `id(client)`
    # since the process-wide client (see `get_mongo_client`) is stable for
    # the life of the process.
    _status_cache: dict[tuple[int, str], tuple[float, dict[str, Any]]] = {}

    def __init__(self, client: MongoClient, collection_name: str = "kill_switch") -> None:
        self._collection: Collection = client[settings.mongo_db_name][collection_name]
        self._cache_key = (id(client), collection_name)

    def status(self) -> dict[str, Any]:
        cached = KillSwitch._status_cache.get(self._cache_key)
        now = time.monotonic()
        if cached is not None and now < cached[0]:
            return dict(cached[1])

        doc = self._collection.find_one({"_id": _DOC_ID})
        if doc is None:
            result = {"killed": False, "reason": None, "updated_at": None, "updated_by": None}
        else:
            result = {
                "killed": doc.get("killed", False),
                "reason": doc.get("reason"),
                "updated_at": doc.get("updated_at"),
                "updated_by": doc.get("updated_by"),
            }
        KillSwitch._status_cache[self._cache_key] = (now + _CACHE_TTL_SECONDS, result)
        return dict(result)

    def set_killed(self, killed: bool, reason: str | None, actor: str | None) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        doc = {"killed": killed, "reason": reason, "updated_at": now, "updated_by": actor}
        self._collection.update_one({"_id": _DOC_ID}, {"$set": doc}, upsert=True)
        KillSwitch._status_cache.pop(self._cache_key, None)
        return self.status()

    def check(self) -> None:
        """Raise `AgentKilledError` if the kill switch is currently engaged."""
        current = self.status()
        if current["killed"]:
            raise AgentKilledError(current["reason"])
