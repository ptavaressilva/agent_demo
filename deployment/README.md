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
- An MCP web-search provider reachable from the container (e.g. a Brave API
  key, or swap `MCP_WEB_SEARCH_COMMAND`/`ARGS` for a hosted MCP server over
  `streamable_http` instead of stdio if you'd rather not run `npx` in the
  container -- see `agent_demo/tools/mcp_tools.py`).
- Arize AX space ID + API key, if you want tracing (`ARIZE_SPACE_ID`,
  `ARIZE_API_KEY`).

## 2. Anthropic credentials via AgentCore Identity (recommended for prod)

Rather than baking `ANTHROPIC_API_KEY` into the container/environment, store
it as an AgentCore Identity API-key credential provider and let
`agent_demo/llm.py` resolve it per-call (`ANTHROPIC_AUTH_MODE=agentcore_identity`,
already set in `deployment/Dockerfile`). The starter-toolkit CLI only wraps
OAuth2 providers, not API-key ones, so create it directly via boto3:

```python
import boto3

client = boto3.client("bedrock-agentcore-control", region_name="us-east-1")
client.create_api_key_credential_provider(
    name="anthropic-api-key",  # must match BEDROCK_AGENTCORE_MODEL_PROVIDER_API_KEY_NAME
    apiKey="sk-ant-...",
)
```

Set `BEDROCK_AGENTCORE_MODEL_PROVIDER_API_KEY_NAME` on the Runtime agent to
whatever `name` you used (defaults to `anthropic-api-key`, matching the
example above). The agent's execution role needs permission to read this
credential provider at runtime.

If you'd rather keep it simple for a first deployment, leave
`ANTHROPIC_AUTH_MODE=env` and set `ANTHROPIC_API_KEY` as a plain Runtime
environment variable instead -- `agent_demo/llm.py` supports both.

## 3. Configure and deploy

```sh
# From the repo root. Detects main.py as the entrypoint.
agentcore configure --entrypoint main.py --name job-search-agent

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
agentcore invoke '{"message": "Find me senior backend roles in Lisbon", \
  "candidate_id": "cand_123", \
  "candidate_profile": "8 years backend, Python/Go, prefers remote-first EU companies"}'
```

Response shape matches `main.py`'s `invoke()`:
`{"session_id": ..., "reply": ..., "message_count": ...}`. Pass the returned
`session_id` back in on the next call to continue the same conversation
(the checkpointer resumes short-term memory for that thread).

## 5. Local dev without AgentCore at all

You don't need any of the above to iterate on the agent itself:

```sh
docker compose up -d          # local Mongo + Postgres
uv run python -m main         # serves /invocations and /ping on :8080, same
                               # contract AgentCore Runtime uses in prod
```

or call `agent_demo.runner.run(...)` directly/in a test, bypassing the HTTP
layer entirely -- see `scripts/run_local.py`.
