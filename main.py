"""AWS Bedrock AgentCore entrypoint.

This is a thin adapter: all real logic lives in `agent_demo.runner.run`,
which is plain async Python testable without AgentCore. Deploy with the
`bedrock-agentcore-starter-toolkit` CLI (`agentcore configure` /
`agentcore launch`) -- see deployment/README.md. Run locally for dev/testing
with `uv run main.py`, which serves the same `/invocations` and `/ping`
routes on http://localhost:8080 that AgentCore Runtime calls in production.
"""

from __future__ import annotations

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from agent_demo.runner import InvokeResult, run

app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload: dict) -> dict:
    """payload: {"message": str, "buyer_id": str, "buyer_profile":
    str, "session_id"?: str} -- see agent_demo.runner.InvokeRequest."""
    result: InvokeResult = await run(payload)
    return {
        "session_id": result.session_id,
        "reply": result.reply,
        "message_count": result.message_count,
    }


if __name__ == "__main__":
    app.run()
