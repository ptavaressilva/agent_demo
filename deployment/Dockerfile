# AgentCore Runtime container. Follows the layout the
# `bedrock-agentcore-starter-toolkit` itself generates for a LangGraph agent
# (uv-managed deps, non-root user, CMD runs the entrypoint module directly)
# so it stays compatible with `agentcore launch`'s expectations.
#
# Build from the repo root:
#   docker build -f deployment/Dockerfile -t agent-demo .
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 \
    ANTHROPIC_AUTH_MODE=agentcore_identity

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY main.py ./
COPY agent_demo ./agent_demo
COPY db ./db

RUN uv sync --frozen --no-dev

RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 8080

CMD ["uv", "run", "python", "-m", "main"]
