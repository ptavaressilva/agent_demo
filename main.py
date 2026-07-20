"""AWS Bedrock AgentCore entrypoint.

This is a thin adapter: all real logic lives in `agent_demo.runner.run`,
which is plain async Python testable without AgentCore. Deploy with the
`bedrock-agentcore-starter-toolkit` CLI (`agentcore configure` /
`agentcore launch`) -- see deployment/README.md. Run locally for dev/testing
with `uv run main.py`, which serves the same `/invocations` and `/ping`
routes on http://localhost:8080 that AgentCore Runtime calls in production.

Kill switch: two ways to reach it, sharing one dispatcher
(`_dispatch_kill_switch_action`):

- `/admin/kill-switch` (GET status, POST to toggle) -- a plain Starlette
  route appended directly to the app, since `BedrockAgentCoreApp` only
  registers `/invocations`/`/ping`/`/ws` itself. Works for local dev and any
  deployment reachable over plain HTTP (e.g. behind your own ALB), but is
  NOT reachable through the managed AgentCore Runtime invoke path
  (`agentcore invoke` / `InvokeAgentRuntime`), which only calls
  `/invocations`.
- A `_kill_switch_admin_action` key inside the `/invocations` payload itself
  -- mirrors the SDK's own `_agent_core_app_action` payload-key convention
  for its built-in debug actions (see `BedrockAgentCoreApp._handle_task_action`).
  Since the payload body is the one thing guaranteed to cross the managed
  invoke path unmodified (custom headers are not), auth here travels in the
  body as a `token` field rather than a header. This is the one to use when
  deployed exclusively behind managed AgentCore Runtime.

Either way, enforcement (`run()` refusing to act while killed) works
identically on both the managed invoke path and plain HTTP -- only the
*toggle* surface differs in reachability.
"""

from __future__ import annotations

import asyncio
import hmac

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from agent_demo.config import settings
from agent_demo.kill_switch import AgentKilledError, KillSwitch
from agent_demo.runner import InvokeResult, get_mongo_client, run

app = BedrockAgentCoreApp()

_ADMIN_ACTION_KEY = "_kill_switch_admin_action"


@app.entrypoint
async def invoke(payload: dict) -> dict | Response:
    """payload: {"message": str, "buyer_id": str, "buyer_profile":
    str, "session_id"?: str} -- or, to resume a run paused for human
    approval, {"session_id": str, "buyer_id": str, "resume_decision": dict}.
    See agent_demo.runner.InvokeRequest.

    A payload containing `_kill_switch_admin_action` is instead routed to
    the in-band kill-switch admin action (see module docstring) and never
    reaches `run()`/the normal buyer flow."""
    if isinstance(payload, dict) and _ADMIN_ACTION_KEY in payload:
        return await _handle_kill_switch_admin_action(payload[_ADMIN_ACTION_KEY])

    try:
        result: InvokeResult = await run(payload)
    except AgentKilledError as e:
        return JSONResponse({"error": "agent_disabled", "reason": e.reason}, status_code=503)
    return {
        "session_id": result.session_id,
        "reply": result.reply,
        "message_count": result.message_count,
        "pending_approval": result.pending_approval,
    }


def _serialize_status(status: dict) -> dict:
    updated_at = status["updated_at"]
    return {**status, "updated_at": updated_at.isoformat() if updated_at else None}


async def _dispatch_kill_switch_action(token: object, body: dict) -> tuple[dict, int]:
    """Shared core for both the HTTP admin route and the in-band
    `/invocations` action: validate the token, then read or set the kill
    switch. `body` may include `killed` (bool, optional -- omit to just read
    status), `reason` (str, optional), `actor` (str, optional). Returns
    (json_body, http_status_code).

    Fails closed: an unset `KILL_SWITCH_ADMIN_TOKEN` is a 500 (misconfigured),
    never treated as "no auth required".
    """
    if not settings.kill_switch_admin_token:
        return {
            "error": "Kill switch admin endpoint is not configured (set KILL_SWITCH_ADMIN_TOKEN)."
        }, 500

    if not isinstance(token, str) or not hmac.compare_digest(
        token.encode("utf-8"), settings.kill_switch_admin_token.encode("utf-8")
    ):
        return {"error": "Unauthorized"}, 401

    kill_switch = KillSwitch(get_mongo_client())

    if "killed" not in body:
        status = await asyncio.to_thread(kill_switch.status)
        return _serialize_status(status), 200

    killed = body.get("killed")
    if not isinstance(killed, bool):
        return {"error": "'killed' must be a bool; omit it entirely to just read status"}, 400

    reason = body.get("reason")
    if reason is not None and not isinstance(reason, str):
        return {"error": "'reason' must be a string"}, 400

    actor = body.get("actor")
    if actor is not None and not isinstance(actor, str):
        return {"error": "'actor' must be a string"}, 400

    status = await asyncio.to_thread(kill_switch.set_killed, killed, reason, actor)
    return _serialize_status(status), 200


async def _admin_kill_switch(request: Request) -> Response:
    """GET returns the current status. POST {"killed": bool, "reason"?: str,
    "actor"?: str} toggles it -- or, with `killed` omitted, also just reads
    status (matching the in-band action, which has no GET/POST distinction).
    Both require a matching X-Admin-Token header.

    Runs on the app's main event loop (custom routes bypass AgentCore's
    worker-loop dispatch for the main entrypoint), so the underlying
    sync-pymongo calls in `_dispatch_kill_switch_action` are pushed to a
    thread to avoid blocking /ping.
    """
    token = request.headers.get("X-Admin-Token", "")

    if request.method == "GET":
        body: dict = {}
    else:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)

    json_body, status_code = await _dispatch_kill_switch_action(token, body)
    return JSONResponse(json_body, status_code=status_code)


async def _handle_kill_switch_admin_action(action: object) -> Response:
    """In-band equivalent of `_admin_kill_switch`, reached via a
    `_kill_switch_admin_action` key in the `/invocations` payload instead of
    a separate route+header, since that's what actually crosses the managed
    AgentCore Runtime invoke path. `action`: {"token": str, "killed"?: bool,
    "reason"?: str, "actor"?: str}."""
    if not isinstance(action, dict):
        return JSONResponse(
            {"error": f"'{_ADMIN_ACTION_KEY}' must be an object"}, status_code=400
        )

    json_body, status_code = await _dispatch_kill_switch_action(action.get("token"), action)
    return JSONResponse(json_body, status_code=status_code)


app.routes.append(Route("/admin/kill-switch", _admin_kill_switch, methods=["GET", "POST"]))


if __name__ == "__main__":
    app.run()
