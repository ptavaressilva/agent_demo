"""Invoke an agent directly (no HTTP, no AgentCore) for local dev/testing.

Requires local Mongo (+ Postgres, for house-search) via `docker compose up
-d` and a real LLM_GATEWAY_API_KEY / ANTHROPIC_API_KEY in .env.

Usage:
    uv run python scripts/run_local.py "Find me a 3-bedroom house in Lisbon"
    AGENT_ID=faq-agent uv run python scripts/run_local.py "What does the kill switch do?"
"""

from __future__ import annotations

import asyncio
import sys

from agent_demo.platform import harness
from agent_demo.platform.registry import load_agent

BUYER_PROFILE = """\
Household of three (couple + one child) looking for a first home.
Wants 3+ bedrooms, a garden or outdoor space, and good access to public
transit. Prefers Lisbon or nearby commuter towns. Budget ceiling: EUR
450,000. Open to needing light renovation but not a full fixer-upper.
"""


def _build_payload(agent_id: str, message: str) -> dict:
    if agent_id == "house-search":
        return {
            "message": message,
            "buyer_id": "local-dev-buyer",
            "buyer_profile": BUYER_PROFILE,
        }
    return {"message": message}


async def main() -> None:
    agent = load_agent()
    message = " ".join(sys.argv[1:]) or "Find me a few houses that fit my profile."
    result = await harness.run(agent, _build_payload(agent.agent_id, message))
    print(f"\n--- session_id: {result.session_id} ({result.message_count} messages) ---\n")
    print(result.reply)


if __name__ == "__main__":
    asyncio.run(main())
