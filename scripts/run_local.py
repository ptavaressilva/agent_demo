"""Invoke the agent directly (no HTTP, no AgentCore) for local dev/testing.

Requires local Mongo + Postgres (`docker compose up -d`) and a real
ANTHROPIC_API_KEY in .env. Optionally set BRAVE_API_KEY for real web search
results (falls back to the MCP server's unauthenticated/degraded behavior
without it, if it has one).

Usage:
    uv run python scripts/run_local.py "Find me senior backend roles in Lisbon"
"""

from __future__ import annotations

import asyncio
import sys

from agent_demo.runner import run

CANDIDATE_PROFILE = """\
8 years of backend engineering experience, primarily Python and Go.
Strong in distributed systems and API design. Prefers remote-first
companies based in Europe. Target roles: Senior/Staff Backend Engineer.
Salary floor: EUR 90,000.
"""


async def main() -> None:
    message = " ".join(sys.argv[1:]) or "Find me a few senior backend roles that fit my profile."
    result = await run(
        {
            "message": message,
            "candidate_id": "local-dev-candidate",
            "candidate_profile": CANDIDATE_PROFILE,
        }
    )
    print(f"\n--- session_id: {result.session_id} ({result.message_count} messages) ---\n")
    print(result.reply)


if __name__ == "__main__":
    asyncio.run(main())
