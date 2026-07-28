"""Backstop for the registry boundary: onboarding a new agent should mean
editing `agent_demo/agents/manifest.py` only, never anything under
`agent_demo/platform/` (see platform/registry.py's docstring). Nothing at
import time stops a future edit from reaching back into `agent_demo.agents.*`
from elsewhere in the platform package, so this greps for it instead.

`platform/registry.py` is the one allowed exception, and even it may only
import the manifest -- never a specific agent's package/class -- so the
platform truly never has to name an agent to serve it.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PLATFORM_DIR = _REPO_ROOT / "agent_demo" / "platform"
_AGENTS_IMPORT = re.compile(r"^\s*(?:from|import)\s+agent_demo\.agents\b")
_MANIFEST_IMPORT = re.compile(r"^\s*(?:from|import)\s+agent_demo\.agents\.manifest\b")


def test_only_registry_imports_agents_and_only_the_manifest():
    offenders = []
    for path in _PLATFORM_DIR.rglob("*.py"):
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not _AGENTS_IMPORT.search(line):
                continue
            if path.name == "registry.py" and _MANIFEST_IMPORT.search(line):
                continue
            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Found a reference to agent_demo.agents.* under agent_demo/platform/ "
        "outside registry.py's import of the manifest -- the platform "
        "should never have to name a specific agent to serve it. Add new "
        "agents to agent_demo/agents/manifest.py instead:\n"
        + "\n".join(offenders)
    )
