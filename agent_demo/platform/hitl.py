"""Standard envelope for a tool that must pause for human approval before
taking a side effect, wrapping LangGraph's `interrupt()` primitive.

This is the one seam in the platform that isn't fully structural: LangGraph's
`interrupt()` is called from inside a tool function's body, and there is no
entrypoint-level wrapper that can intercept or require that call site the way
the kill switch/tracing/gateway routing are enforced. Every agent tool that
needs human approval should call `request_approval`/`is_approved` here
instead of `langgraph.types.interrupt` directly, so the approve/reject
envelope shape is consistent across agents -- backstopped by
`tests/test_no_raw_interrupt.py`, which fails CI if a raw `interrupt(` call
shows up in agent tool code outside this module.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt


def request_approval(action: str, payload: dict[str, Any], message: str) -> Any:
    """Pause the graph for human review. `payload` are the tool's proposed
    arguments; `message` is shown to the reviewer. Returns whatever the
    resume call supplies as `resume_decision` -- typically a dict with an
    `action` key (`"approve"`/`"reject"`) and optionally edited fields."""
    return interrupt({"action": action, **payload, "message": message})


def is_approved(decision: Any) -> bool:
    return isinstance(decision, dict) and decision.get("action") == "approve"
