"""Kill switch: an operator-controlled stop for the agent, checked at the
start of every invocation -- including resumes of a paused human-in-the-loop
approval -- so an incident can be contained without a redeploy.

State lives in Mongo (`kill_switch` collection, single `_id="global"`
document) rather than an env var or in-process flag: it's shared correctly
across every running instance and survives restarts, so flipping it during
an incident takes effect immediately for all traffic instead of only after a
redeploy.

Scope: this gates entry to a new turn/resume. It does not interrupt a turn
that's already mid-run -- there's no per-step check inside the ReAct loop.
That's bounded anyway by `max_react_steps`, so a single in-flight run
finishing is an acceptable tradeoff for not adding a check on every graph
step.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection

from agent_demo.config import settings

_DOC_ID = "global"


class AgentKilledError(Exception):
    """Raised by `run()` when the kill switch is engaged. Carries the
    operator-supplied reason so callers/logs can surface why."""

    def __init__(self, reason: str | None) -> None:
        self.reason = reason
        super().__init__(reason or "Agent is currently disabled by the kill switch.")


class KillSwitch:
    def __init__(self, client: MongoClient, collection_name: str = "kill_switch") -> None:
        self._collection: Collection = client[settings.mongo_db_name][collection_name]

    def status(self) -> dict[str, Any]:
        doc = self._collection.find_one({"_id": _DOC_ID})
        if doc is None:
            return {"killed": False, "reason": None, "updated_at": None, "updated_by": None}
        return {
            "killed": doc.get("killed", False),
            "reason": doc.get("reason"),
            "updated_at": doc.get("updated_at"),
            "updated_by": doc.get("updated_by"),
        }

    def set_killed(self, killed: bool, reason: str | None, actor: str | None) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        doc = {"killed": killed, "reason": reason, "updated_at": now, "updated_by": actor}
        self._collection.update_one({"_id": _DOC_ID}, {"$set": doc}, upsert=True)
        return self.status()

    def check(self) -> None:
        """Raise `AgentKilledError` if the kill switch is currently engaged."""
        current = self.status()
        if current["killed"]:
            raise AgentKilledError(current["reason"])
