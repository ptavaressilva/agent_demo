"""Invoke the agent directly (no HTTP, no AgentCore) for local dev/testing.

Requires local Mongo + Postgres (`docker compose up -d`) and a real
ANTHROPIC_API_KEY in .env. Optionally set BRAVE_API_KEY for real web search
results (falls back to the MCP server's unauthenticated/degraded behavior
without it, if it has one).

Usage:
    uv run python scripts/run_local.py "Find me a 3-bedroom house in Lisbon"
"""

from __future__ import annotations

import asyncio
import sys

from agent_demo.runner import run

BUYER_PROFILE = """\
Household of three (couple + one child) looking for a first home.
Wants 3+ bedrooms, a garden or outdoor space, and good access to public
transit. Prefers Lisbon or nearby commuter towns. Budget ceiling: EUR
450,000. Open to needing light renovation but not a full fixer-upper.
"""


async def main() -> None:
    message = " ".join(sys.argv[1:]) or "Find me a few houses that fit my profile."
    result = await run(
        {
            "message": message,
            "buyer_id": "local-dev-buyer",
            "buyer_profile": BUYER_PROFILE,
        }
    )
    print(f"\n--- session_id: {result.session_id} ({result.message_count} messages) ---\n")
    print(result.reply)


if __name__ == "__main__":
    asyncio.run(main())
