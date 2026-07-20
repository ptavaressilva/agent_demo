# Deploying to AWS Bedrock AgentCore Runtime

This has **not** been run against AWS as part of building this repo (per
scope). Everything below was verified against the installed
`bedrock-agentcore` / `bedrock-agentcore-starter-toolkit` packages and the
`boto3` `bedrock-agentcore-control` client, but the actual `deploy` step
needs your AWS account/credentials and hasn't been executed.

> **Toolkit note:** `bedrock-agentcore-starter-toolkit` (the Python CLI used
> below, `agentcore ...`) prints a deprecation notice pointing at a newer
> Node-based `@aws/agentcore` CLI (`npm install -g @aws/agentcore`). The
> Python toolkit is what's installed in this repo's dev dependencies and is
> what the steps below use; if you have the newer CLI available, prefer it
> and adapt these steps to its command names.

## 1. One-time AWS setup

- An ECR repository (or let `agentcore configure` create one).
- An IAM execution role for the AgentCore Runtime agent (or let `configure`
  create one) with permissions for: pulling from ECR, `bedrock:InvokeModel*`
  if you ever want to route through Bedrock instead of the direct Anthropic
  API, and CloudWatch Logs.
- Network reachability from the Runtime's VPC config to your MongoDB and
  Postgres instances (Atlas / RDS / self-hosted) -- the `docker-compose.yml`
  Mongo/Postgres in this repo are for **local dev only**, not what the
  deployed agent talks to. Point `MONGO_URI` / `POSTGRES_DSN` at your real
  instances via the Runtime's environment configuration.
- A reachable LLM gateway (LiteLLM proxy) -- the `litellm` service in
  `docker-compose.yml` is local-dev only. In production, run LiteLLM as its
  own service (its own deployment, or LiteLLM's hosted offering) and point
  `LLM_GATEWAY_BASE_URL` at it via the Runtime's environment configuration.
  See "Anthropic credentials" below for how the gateway key itself is kept
  out of the container.
- An MCP web-search provider reachable from the container (e.g. a Brave API
  key, or swap `MCP_WEB_SEARCH_COMMAND`/`ARGS` for a hosted MCP server over
  `streamable_http` instead of stdio if you'd rather not run `npx` in the
  container -- see `agent_demo/lib/mcp_tools.py`).
- Arize AX space ID + API key, if you want tracing (`ARIZE_SPACE_ID`,
  `ARIZE_API_KEY`).

## 2. LLM gateway credentials via AgentCore Identity (recommended for prod)

The deployed agent never talks to Anthropic directly -- it calls an LLM
gateway (LiteLLM proxy) over `LLM_GATEWAY_BASE_URL`, and the *gateway*
(not this container) holds `ANTHROPIC_API_KEY`. The only secret this
container needs is the gateway's own key. Rather than baking
`LLM_GATEWAY_API_KEY` into the container/environment, store it as an
AgentCore Identity API-key credential provider and let
`agent_demo/platform/llm.py` resolve it per-call (`LLM_GATEWAY_AUTH_MODE=agentcore_identity`, already set
in `deployment/Dockerfile`). The starter-toolkit CLI only wraps OAuth2
providers, not API-key ones, so create it directly via boto3:

```python
import boto3

client = boto3.client("bedrock-agentcore-control", region_name="us-east-1")
client.create_api_key_credential_provider(
    name="llm-gateway-api-key",  # must match BEDROCK_AGENTCORE_MODEL_PROVIDER_API_KEY_NAME
    apiKey="sk-...",  # the gateway's LiteLLM master/virtual key, not the Anthropic key
)
```

Set `BEDROCK_AGENTCORE_MODEL_PROVIDER_API_KEY_NAME` on the Runtime agent to
whatever `name` you used (defaults to `llm-gateway-api-key`, matching the
example above). The agent's execution role needs permission to read this
credential provider at runtime.

If you'd rather keep it simple for a first deployment, leave
`LLM_GATEWAY_AUTH_MODE=env` and set `LLM_GATEWAY_API_KEY` as a plain Runtime
environment variable instead -- `agent_demo/platform/llm.py` supports both. Either
way, also set `LLM_GATEWAY_BASE_URL` to the gateway's real (non-localhost)
endpoint.

## 3. Configure and deploy

```sh
# From the repo root. Detects main.py as the entrypoint.
agentcore configure --entrypoint main.py --name house-search-agent

# Cloud build + deploy (no local Docker required):
agentcore deploy

# Or build/run the container locally first to sanity-check it:
agentcore deploy --local
```

`agentcore configure` generates its own `.bedrock_agentcore.yaml` and
Dockerfile from `main.py` -- `deployment/Dockerfile` in this repo is a
hand-written equivalent (uv-based, same CMD shape) kept for reference /
`docker build` outside the toolkit; if you run `agentcore configure`, let it
regenerate the Dockerfile it needs rather than fighting it over the file.

## 4. Invoke

```sh
agentcore invoke '{"message": "Find me a 3-bedroom house in Lisbon", \
  "buyer_id": "buyer_123", \
  "buyer_profile": "Family of three, budget EUR 450k, wants a garden and good transit access"}'
```

Response shape matches `main.py`'s `invoke()`:
`{"session_id": ..., "reply": ..., "message_count": ...}`. Pass the returned
`session_id` back in on the next call to continue the same conversation
(the checkpointer resumes short-term memory for that thread).

## 5. Kill switch

`agent_demo/platform/kill_switch.py` gates every invocation (new turns and
resumed approvals), for whichever agent this deployment serves, on a flag
stored in Mongo, so an incident can be contained without a redeploy.
`main.py` serves it at `/admin/kill-switch`:

```sh
# Check status
curl -H "X-Admin-Token: $KILL_SWITCH_ADMIN_TOKEN" http://localhost:8080/admin/kill-switch

# Engage it
curl -X POST -H "X-Admin-Token: $KILL_SWITCH_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"killed": true, "reason": "investigating runaway tool calls"}' \
  http://localhost:8080/admin/kill-switch

# Release it
curl -X POST -H "X-Admin-Token: $KILL_SWITCH_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"killed": false}' \
  http://localhost:8080/admin/kill-switch
```

Set `KILL_SWITCH_ADMIN_TOKEN` (both toggle surfaces below fail closed if it's
unset).

**`/admin/kill-switch` is a plain HTTP route on the container, not an
AgentCore Runtime action.** It works locally and for any deployment
reachable over plain HTTP (e.g. behind your own ALB). It is *not* reachable
through the managed AgentCore Runtime invoke path (`agentcore invoke` / the
`InvokeAgentRuntime` API), which only exposes `/invocations`.

If you deploy behind that path exclusively, use the in-band equivalent
instead -- a `_kill_switch_admin_action` key inside the `/invocations`
payload itself, which `main.py`'s `invoke()` intercepts before it ever
reaches the normal buyer flow. Auth travels in the body (a `token` field)
rather than a header, since the payload is the one thing guaranteed to cross
the managed invoke path unmodified:

```sh
# Check status
agentcore invoke '{"_kill_switch_admin_action": {"token": "'"$KILL_SWITCH_ADMIN_TOKEN"'"}}'

# Engage it
agentcore invoke '{"_kill_switch_admin_action": {"token": "'"$KILL_SWITCH_ADMIN_TOKEN"'", \
  "killed": true, "reason": "investigating runaway tool calls"}}'

# Release it
agentcore invoke '{"_kill_switch_admin_action": {"token": "'"$KILL_SWITCH_ADMIN_TOKEN"'", "killed": false}}'
```

Both surfaces share the same underlying `KillSwitch`/Mongo state, so either
one can release a switch the other engaged. As a last resort (e.g. the
token is lost), a direct write to the `kill_switch` Mongo collection also
works (`KillSwitch(client).set_killed(...)`).

Engaging the switch blocks new turns/resumes (via either surface); it does
not abort a run already mid-loop (each run is bounded by `max_react_steps`
regardless).

## 6. Local dev without AgentCore at all

You don't need any of the above to iterate on the agent itself:

```sh
docker compose up -d          # local Mongo + Postgres
uv run python -m main         # serves /invocations and /ping on :8080, same
                               # contract AgentCore Runtime uses in prod
```

or call `agent_demo.platform.harness.run(agent, ...)` directly/in a test,
bypassing the HTTP layer entirely -- see `scripts/run_local.py`.
