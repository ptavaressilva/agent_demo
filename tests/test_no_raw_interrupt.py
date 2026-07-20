"""Backstop for the one seam in the platform that isn't fully structural:
`agent_demo.platform.hitl.request_approval` wraps LangGraph's `interrupt()`
to standardize the approve/reject envelope, but nothing at the entrypoint
level can force a tool author to use it instead of calling
`langgraph.types.interrupt` directly (see platform/hitl.py's docstring).

This test greps every agent's tool source for a raw `interrupt(` call
outside `platform/hitl.py` and fails if one shows up -- catching an
agent-side bypass of the shared approval envelope at CI time, even though
nothing prevents it at import/runtime.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_DIR = _REPO_ROOT / "agent_demo" / "agents"
_RAW_INTERRUPT_CALL = re.compile(r"(?<!request_)\binterrupt\s*\(")


def test_no_agent_tool_calls_interrupt_directly():
    offenders = []
    for path in _AGENTS_DIR.rglob("*.py"):
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _RAW_INTERRUPT_CALL.search(line):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Found a raw interrupt( call outside agent_demo.platform.hitl -- use "
        "platform.hitl.request_approval instead so the approve/reject "
        "envelope stays consistent across agents:\n" + "\n".join(offenders)
    )
