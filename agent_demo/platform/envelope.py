"""The generic, agent-agnostic part of an invocation's request/response
shape. Every agent's own request schema (e.g.
`agent_demo.agents.house_search.spec.HouseSearchRequest`) subclasses
`BaseInvokeEnvelope` and adds whatever domain fields it needs -- so the
resume/session/budget-override plumbing is written once here rather than
copy-pasted per agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator


class BaseInvokeEnvelope(BaseModel):
    """`buyer_profile`-style domain fields cross a real trust boundary the
    same way `message` does (arbitrary per-request text from the caller) --
    each agent's own subclass is responsible for the same length/type
    discipline applied to `message`/`resume_decision` here.
    """

    message: str | None = Field(default=None, min_length=1, max_length=8000)
    session_id: str | None = None

    # Optional per-request overrides of the agent's own defaults, clamped to
    # the platform ceiling by `platform.budget.clamp_budget` -- see
    # `platform/harness.py`.
    max_react_steps: int | None = Field(default=None, ge=1, le=50)
    max_self_correction_retries: int | None = Field(default=None, ge=0, le=10)

    # Set to resume a run paused on a human-in-the-loop approval instead of
    # starting a new turn -- e.g. {"action": "approve"} or {"action":
    # "reject"}, optionally with edited fields that override the tool's
    # original arguments. Crosses the same trust boundary as `message`:
    # arbitrary per-request data that, on approval, flows straight into a
    # tool's side effect, so it gets the same length/type check and no more.
    resume_decision: dict[str, str] | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_message_xor_resume(self) -> "BaseInvokeEnvelope":
        if self.resume_decision is not None:
            if not self.session_id:
                raise ValueError("resume_decision requires an existing session_id.")
            if any(len(value) > 20000 for value in self.resume_decision.values()):
                raise ValueError("resume_decision values must each be at most 20000 characters.")
        elif not self.message:
            raise ValueError("message is required unless resuming with resume_decision.")
        return self


@dataclass
class InvokeResult:
    session_id: str
    reply: str
    message_count: int
    # Set when the run paused on a human-in-the-loop approval instead of
    # finishing -- the interrupt's payload (see platform/hitl.py). Resume by
    # calling `harness.run()` again with the same session_id and a
    # resume_decision.
    pending_approval: dict | None = None
