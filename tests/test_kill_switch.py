"""Tests for `KillSwitch`/`AgentKilledError` against a fake Mongo collection
(mirrors the FakePool pattern in test_postgres_tools.py) -- no real Mongo
needed.
"""

from __future__ import annotations

import pytest

from agent_demo.kill_switch import AgentKilledError, KillSwitch

pytestmark = pytest.mark.usefixtures("_clear_kill_switch_cache")


@pytest.fixture
def _clear_kill_switch_cache():
    """`KillSwitch._status_cache` is a class attribute shared across
    instances (it's keyed by `id(client)`, and Python can reuse a
    just-garbage-collected FakeMongoClient's id for a new one within a
    test run) -- clear it around each test so caching in one test can't
    leak into another."""
    KillSwitch._status_cache.clear()
    yield
    KillSwitch._status_cache.clear()


class FakeCollection:
    """Just enough of pymongo's Collection API for KillSwitch: find_one and
    an upserting update_one against a single in-memory doc."""

    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}

    def find_one(self, filter_: dict) -> dict | None:
        return self._docs.get(filter_["_id"])

    def update_one(self, filter_: dict, update: dict, upsert: bool = False) -> None:
        doc_id = filter_["_id"]
        doc = self._docs.setdefault(doc_id, {"_id": doc_id})
        doc.update(update["$set"])


class FakeDB:
    def __init__(self) -> None:
        self._collections: dict[str, FakeCollection] = {}

    def __getitem__(self, name: str) -> FakeCollection:
        return self._collections.setdefault(name, FakeCollection())


class FakeMongoClient:
    def __init__(self) -> None:
        self._dbs: dict[str, FakeDB] = {}

    def __getitem__(self, name: str) -> FakeDB:
        return self._dbs.setdefault(name, FakeDB())


def test_status_defaults_to_not_killed_when_no_document_exists():
    kill_switch = KillSwitch(FakeMongoClient())

    status = kill_switch.status()

    assert status == {"killed": False, "reason": None, "updated_at": None, "updated_by": None}


def test_set_killed_then_status_reflects_it():
    kill_switch = KillSwitch(FakeMongoClient())

    status = kill_switch.set_killed(True, reason="investigating", actor="pedro")

    assert status["killed"] is True
    assert status["reason"] == "investigating"
    assert status["updated_by"] == "pedro"
    assert status["updated_at"] is not None
    assert kill_switch.status() == status


def test_set_killed_can_release_the_switch_again():
    kill_switch = KillSwitch(FakeMongoClient())
    kill_switch.set_killed(True, reason="investigating", actor="pedro")

    status = kill_switch.set_killed(False, reason=None, actor="pedro")

    assert status["killed"] is False
    assert kill_switch.status()["killed"] is False


def test_check_is_a_noop_when_not_killed():
    kill_switch = KillSwitch(FakeMongoClient())

    kill_switch.check()  # must not raise


def test_check_raises_agent_killed_error_with_reason_when_engaged():
    kill_switch = KillSwitch(FakeMongoClient())
    kill_switch.set_killed(True, reason="on fire", actor="pedro")

    with pytest.raises(AgentKilledError) as exc_info:
        kill_switch.check()

    assert exc_info.value.reason == "on fire"


def test_status_is_cached_within_the_ttl():
    kill_switch = KillSwitch(FakeMongoClient())
    calls = 0
    original_find_one = kill_switch._collection.find_one

    def counting_find_one(filter_):
        nonlocal calls
        calls += 1
        return original_find_one(filter_)

    kill_switch._collection.find_one = counting_find_one

    kill_switch.status()
    kill_switch.status()

    assert calls == 1


def test_status_cache_expires_after_the_ttl(monkeypatch):
    import agent_demo.kill_switch as kill_switch_module

    fake_now = [1_000.0]
    monkeypatch.setattr(kill_switch_module.time, "monotonic", lambda: fake_now[0])

    kill_switch = KillSwitch(FakeMongoClient())
    calls = 0
    original_find_one = kill_switch._collection.find_one

    def counting_find_one(filter_):
        nonlocal calls
        calls += 1
        return original_find_one(filter_)

    kill_switch._collection.find_one = counting_find_one

    kill_switch.status()
    fake_now[0] += kill_switch_module._CACHE_TTL_SECONDS + 0.1
    kill_switch.status()

    assert calls == 2


def test_set_killed_invalidates_the_cache_for_immediate_effect():
    kill_switch = KillSwitch(FakeMongoClient())
    kill_switch.status()  # warm the cache with killed=False

    kill_switch.set_killed(True, reason="on fire", actor="pedro")

    assert kill_switch.status()["killed"] is True
