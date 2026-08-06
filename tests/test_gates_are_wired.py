"""The gate scripts run in the workflows that need them.

A gate is only a gate where it is invoked. CLAUDE.md lists seven checks a
contributor runs, three of them scripts under ``scripts/``, and until
2026-08-05 those three ran in exactly one CI job (``encryption`` in ci.yml)
which a tag push does not trigger. So the release workflow, the one that
builds the sdist and wheel that reach PyPI, ran ruff, mypy and pytest and
none of the three, including ``readme-numbers.sh``, which holds what CLAUDE.md
calls the invariant engram has paid for most.

Read as text rather than parsed as YAML on purpose: PyYAML is not a declared
dependency of this project or of its ``dev`` extra, it is present here only
transitively, and a test that quietly depends on somebody else's transitive
install is the same class of defect as a gate that runs in one job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

_GATE_SCRIPTS = (
    "scripts/no-network-at-write.sh",
    "scripts/readme-numbers.sh",
    "scripts/local-first.sh",
)


@pytest.mark.parametrize("script", _GATE_SCRIPTS)
def test_the_release_workflow_runs_every_gate_script(script: str) -> None:
    """A tag is the one push that ships bytes to users, and it does not run
    ci.yml at all: ci.yml triggers on pushes to main/master and on pull
    requests. Whatever holds an invariant has to be here too."""
    release = (_WORKFLOWS / "release.yml").read_text()
    assert script in release, (
        f"{script} does not run in release.yml, so a tag can publish a release that never passed it"
    )


@pytest.mark.parametrize("script", _GATE_SCRIPTS)
def test_ci_still_runs_every_gate_script(script: str) -> None:
    """The release workflow is the addition, not the replacement: a pull
    request has to fail before a tag ever gets cut."""
    ci = (_WORKFLOWS / "ci.yml").read_text()
    assert script in ci, f"{script} no longer runs in ci.yml"


@pytest.mark.parametrize(
    "script",
    # The two that need this project's dependencies: local-first.sh imports
    # engram, readme-numbers.sh runs the suite. no-network-at-write.sh is
    # deliberately absent, because it reads the source with ast and imports
    # nothing of ours; if it ever does, it joins this list.
    ["scripts/local-first.sh", "scripts/readme-numbers.sh"],
)
def test_a_gate_that_needs_our_dependencies_finds_the_interpreter_that_has_them(
    script: str,
) -> None:
    """Hardcoding `python3` runs the system interpreter on any machine whose
    dependencies live in a virtualenv, and the gate then dies on
    ModuleNotFoundError rather than measuring anything. It passed in CI only
    because CI has no .venv, which is the worst version of this: green
    everywhere it is watched, unrunnable everywhere it is written down.
    """
    text = (Path(__file__).resolve().parents[1] / script).read_text()
    assert "[ -x .venv/bin/python ]" in text, (
        f"{script} does not fall back to .venv/bin/python, so a contributor "
        f"following CLAUDE.md's gate list cannot run it"
    )


def test_the_gates_the_docs_name_are_the_gates_that_run() -> None:
    """CLAUDE.md's gate list is the contract a contributor reads. A script
    added there and wired nowhere is a promise; one wired and never written
    down is a surprise."""
    claude_md = (Path(__file__).resolve().parents[1] / "CLAUDE.md").read_text()
    for script in _GATE_SCRIPTS:
        assert f"./{script}" in claude_md, f"{script} runs in CI but CLAUDE.md never names it"
