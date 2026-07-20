"""Tests for main.py's kill-switch surface: the /admin/kill-switch route
(auth, status, toggle) and the /invocations entrypoint's AgentKilledError ->
503 mapping. Runs against a fake Mongo client (see test_kill_switch.py) and
a patched `run`, so no real Mongo/Postgres/MCP/Anthropic is needed.
"""

from __future__ import annotations

import pytest
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

import main
from agent_demo.kill_switch import AgentKilledError
from tests.test_kill_switch import FakeMongoClient


@pytest.fixture
def fake_mongo(monkeypatch):
    client = FakeMongoClient()
    monkeypatch.setattr(main, "get_mongo_client", lambda: client)
    return client


@pytest.fixture
def admin_token(monkeypatch):
    monkeypatch.setattr(main.settings, "kill_switch_admin_token", "s3cret")
    return "s3cret"


def test_admin_endpoint_fails_closed_when_token_not_configured(fake_mongo, monkeypatch):
    monkeypatch.setattr(main.settings, "kill_switch_admin_token", "")
    client = TestClient(main.app)

    response = client.get("/admin/kill-switch", headers={"X-Admin-Token": "anything"})

    assert response.status_code == 500


def test_admin_endpoint_rejects_missing_or_wrong_token(fake_mongo, admin_token):
    client = TestClient(main.app)

    assert client.get("/admin/kill-switch").status_code == 401
    assert client.get("/admin/kill-switch", headers={"X-Admin-Token": "wrong"}).status_code == 401


async def test_invoke_admin_action_rejects_non_ascii_token_without_crashing(fake_mongo, admin_token):
    """hmac.compare_digest raises TypeError on non-ASCII str input -- a
    non-ASCII token must fail auth cleanly (401), not surface as a 500. Only
    testable via the in-band action: the HTTP route's httpx test client
    refuses to even send a non-ASCII header value, so this path (arbitrary
    Unicode in a JSON body) is the one that actually exercises the fix."""
    result = await main.invoke({"_kill_switch_admin_action": {"token": "café", "killed": True}})

    assert result.status_code == 401


def test_admin_get_returns_default_status_when_not_killed(fake_mongo, admin_token):
    client = TestClient(main.app)

    response = client.get("/admin/kill-switch", headers={"X-Admin-Token": admin_token})

    assert response.status_code == 200
    assert response.json() == {"killed": False, "reason": None, "updated_at": None, "updated_by": None}


def test_admin_post_toggles_and_get_reflects_it(fake_mongo, admin_token):
    client = TestClient(main.app)
    headers = {"X-Admin-Token": admin_token}

    post_response = client.post(
        "/admin/kill-switch", json={"killed": True, "reason": "incident"}, headers=headers
    )
    assert post_response.status_code == 200
    assert post_response.json()["killed"] is True
    assert post_response.json()["reason"] == "incident"

    get_response = client.get("/admin/kill-switch", headers=headers)
    assert get_response.json()["killed"] is True


def test_admin_post_without_killed_field_just_reads_status(fake_mongo, admin_token):
    """`killed` is optional on POST too -- its presence, not the HTTP verb,
    is what distinguishes a status read from a toggle. This keeps the HTTP
    route and the in-band /invocations action (which has no GET/POST
    distinction at all) behaving identically."""
    client = TestClient(main.app)

    response = client.post("/admin/kill-switch", json={}, headers={"X-Admin-Token": admin_token})

    assert response.status_code == 200
    assert response.json()["killed"] is False


def test_admin_post_rejects_non_bool_killed_field(fake_mongo, admin_token):
    client = TestClient(main.app)

    response = client.post(
        "/admin/kill-switch", json={"killed": "true"}, headers={"X-Admin-Token": admin_token}
    )

    assert response.status_code == 400


async def test_invoke_maps_agent_killed_error_to_503(monkeypatch):
    async def _raise_killed(payload):
        raise AgentKilledError("on fire")

    monkeypatch.setattr(main, "run", _raise_killed)

    result = await main.invoke({"message": "hi", "buyer_id": "b1", "buyer_profile": "x"})

    assert isinstance(result, JSONResponse)
    assert result.status_code == 503
    import json

    assert json.loads(result.body) == {"error": "agent_disabled", "reason": "on fire"}


# --- In-band admin action via /invocations (for deployments only reachable
# through the managed AgentCore invoke path, where /admin/kill-switch is not
# reachable) -- same dispatcher as the HTTP route, just fed from the
# payload's `_kill_switch_admin_action` key instead of headers.


async def test_invoke_routes_admin_action_without_touching_run(fake_mongo, admin_token, monkeypatch):
    async def _boom(payload):
        raise AssertionError("run() should not be called for an admin-action payload")

    monkeypatch.setattr(main, "run", _boom)

    result = await main.invoke(
        {"_kill_switch_admin_action": {"token": admin_token, "killed": True, "reason": "incident"}}
    )

    assert isinstance(result, JSONResponse)
    import json

    body = json.loads(result.body)
    assert result.status_code == 200
    assert body["killed"] is True
    assert body["reason"] == "incident"


async def test_invoke_admin_action_get_status_omits_killed_field(fake_mongo, admin_token):
    result = await main.invoke({"_kill_switch_admin_action": {"token": admin_token}})

    import json

    assert result.status_code == 200
    assert json.loads(result.body)["killed"] is False


async def test_invoke_admin_action_rejects_wrong_token(fake_mongo, admin_token):
    result = await main.invoke({"_kill_switch_admin_action": {"token": "wrong", "killed": True}})

    assert result.status_code == 401


async def test_invoke_admin_action_fails_closed_when_token_not_configured(fake_mongo, monkeypatch):
    monkeypatch.setattr(main.settings, "kill_switch_admin_token", "")

    result = await main.invoke({"_kill_switch_admin_action": {"token": "anything", "killed": True}})

    assert result.status_code == 500


async def test_invoke_admin_action_can_release_the_switch_it_engaged(fake_mongo, admin_token):
    """The in-band action must be able to toggle both directions on its own
    -- it's the only toggle surface for a deployment behind managed
    AgentCore Runtime, so it can't depend on the HTTP route to undo it."""
    await main.invoke({"_kill_switch_admin_action": {"token": admin_token, "killed": True}})
    released = await main.invoke({"_kill_switch_admin_action": {"token": admin_token, "killed": False}})

    import json

    assert json.loads(released.body)["killed"] is False


async def test_invoke_admin_action_payload_never_reaches_invoke_request_validation(fake_mongo, admin_token):
    """A payload with only `_kill_switch_admin_action` has no `message`/
    `buyer_id` -- proves it's short-circuited before InvokeRequest
    validation rather than falling through to run()."""
    result = await main.invoke({"_kill_switch_admin_action": {"token": admin_token}})

    assert result.status_code == 200
