"""Tests for `agent_demo.platform.registry`: the lookup built from
`agent_demo.agents.manifest.AGENTS`. The one invariant worth guarding here is
that `DEFAULT_AGENT_ID` actually resolves -- a manifest edit that renames or
drops the default agent would otherwise pass ceiling validation at import
time and only surface as an `Unknown AGENT_ID` error on the first
`AGENT_ID`-less request.
"""

from __future__ import annotations

import pytest

from agent_demo.agents.manifest import DEFAULT_AGENT_ID
from agent_demo.platform.registry import _REGISTRY, load_agent


def test_default_agent_id_is_registered():
    assert DEFAULT_AGENT_ID in _REGISTRY


def test_load_agent_defaults_to_default_agent_id(monkeypatch):
    monkeypatch.delenv("AGENT_ID", raising=False)
    assert load_agent().agent_id == DEFAULT_AGENT_ID


def test_load_agent_rejects_unknown_id():
    with pytest.raises(RuntimeError, match="Unknown AGENT_ID"):
        load_agent("not-a-registered-agent")
