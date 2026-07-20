"""Step/retry budget clamping shared by every agent through `harness.run`.

An agent picks its own default, and a caller may request a per-turn override
(e.g. a "quick look" vs. a "thorough" run) -- but no override, however large,
can push the effective value past the platform-wide ceiling. This is the
per-request enforcement point; `graph_factory`'s `recursion_limit` is the
structural backstop underneath it (see platform/harness.py).
"""

from __future__ import annotations


def clamp_budget(*, requested: int | None, agent_default: int, platform_ceiling: int) -> int:
    value = requested if requested is not None else agent_default
    return min(value, platform_ceiling)
