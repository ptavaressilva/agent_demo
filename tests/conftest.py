"""`KillSwitch._status_cache` is a class attribute keyed by `id(client)`, and
Python can reuse a just-garbage-collected FakeMongoClient's id for a new one
in a different test -- clear it around every test, globally, so caching in
one test/module can never leak into another regardless of test order.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_kill_switch_cache():
    from agent_demo.platform.kill_switch import KillSwitch

    KillSwitch._status_cache.clear()
    yield
    KillSwitch._status_cache.clear()
