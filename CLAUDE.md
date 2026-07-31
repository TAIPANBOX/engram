# CLAUDE.md, working instructions for engram

These instructions apply to any model working in this repo. Read this file
before starting a task. It holds process and invariants only: **no status.**
Status goes stale, and a stale instruction file is worse than none. Where the
code is: `CHANGELOG.md`. What has actually been measured: `VALIDATION.md`.

## Read before you change anything

1. **`DESIGN.md`, before any non-trivial task.** Do not reinvent a decision
   that is already made there. If it is unclear or self-contradictory, **ask
   rather than guess**.
2. `DATA_FLOW.md` for how a memory moves through the system.
3. `VALIDATION.md` before touching any number that appears in the README, the
   docs, or a release note. See invariant 2, which is the one this project has
   paid for most.

## What engram is

An embeddable cognitive memory layer for AI agents. The "SQLite of agent
memory": a single-file, local-first store modelling episodic, semantic and
procedural memory with temporal validity, importance decay,
spreading-activation retrieval and provenance tracking.

## Stack

- Python 3.11+, packaged with hatchling (PEP 621, `pyproject.toml`)
- SQLite plus the `sqlite-vec` extension, one file per store
- Embeddings via `fastembed` (ONNX, local, no heavy dependencies)
- Reflection LLM pluggable (Anthropic, OpenAI, Ollama), never required to write
- ruff for lint and format, mypy `--strict` for types, pytest for tests

## The working loop

1. **Plan first** for anything touching multiple modules or making an
   architectural or dependency decision. Write the plan, get the user's
   agreement, then implement. Small single-file fixes can skip it.
2. Implement one logical increment. Match surrounding style; comment only where
   the *why* is non-obvious.
3. Run every gate below. All must pass before you call anything done.
4. Commit small, one logical change, Conventional Commits. End the message with
   the standard co-author trailer naming the model that actually did the work.
5. Open a PR with `gh`, wait for CI to be green.
6. **Ask the user before merging.** Do not self-merge.

## Gates

```sh
ruff check .
ruff format --check .
mypy engram
pytest
./scripts/no-network-at-write.sh
```

CI additionally runs the encryption suite on its own
(`pytest tests/test_encryption.py -v`) and `pip-audit`. Note `ruff format
--check`, not `ruff format`: CI checks formatting rather than applying it, so a
local run that reformats and then passes is not the same signal.

## Hard invariants

Each one carries how it is held today. Use `(gate: ...)`, `(test: ...)`,
`(partly gated: ...)` or `(not enforced)`, and use the weakest one that is
true. An invariant with no check, written as though it had one, is worse than
an absent invariant.

1. **No network call at write time.** Storing a memory is local, always. Only
   `reflect()` may reach an external LLM, and only when the caller asks for it.
   A provider SDK imported at module level would make an optional capability a
   hard requirement and put a network dependency behind an import.
   *(gate: `scripts/no-network-at-write.sh`)*
2. **Every number this project publishes is one somebody measured, and the
   command that measured it is written down next to it.** A number in a README
   is a claim with no owner. This is the invariant engram has paid for most:
   2.4.1 exists only because the headline table still described the previous
   blend weights one release after they stopped being the default, and nobody
   could have caught that from the table alone. If you change a default, a
   weight, or an algorithm, **the numbers describing it are part of the same
   change**, not a follow-up. *(not enforced)*
3. **Recall is measured on the full public benchmark, not on a fixture we
   wrote.** A benchmark we author ourselves measures agreement with our own
   assumptions. Where a metric is knowingly an upper or lower bound, say so
   next to it rather than rounding the caveat away. *(not enforced)*
4. **Local-first is a constraint, not a preference.** A single file you can
   copy, with no service to run. A dependency that needs a server, or an
   embedding path that requires a network round trip, breaks the one sentence
   that describes this project. *(not enforced)*
5. **No global state.** Pass the `Engram` instance explicitly. Two stores in
   one process must not be able to see each other. *(not enforced)*
6. **Public APIs are strictly typed and `mypy --strict` passes.**
   *(gate: `mypy engram` in CI and in the gate list above)*
7. **One module, one concern.** No grab-bag files. *(not enforced)*
8. **Every new function gets a test.** *(not enforced)*

## Decisions that have no gate yet

This list is debt, and it is here to stay visible rather than to be tidy.

**Held by this file alone: invariants 2, 3, 4, 5, 7 and 8.**

**Invariant 2 is the one to automate, and it is worth more here than anywhere
else in the estate**, because this project's history is a sequence of releases
fixing published numbers. The shape that works is a script that extracts every
number from the README and the docs which claims to be a measurement, looks it
up in `VALIDATION.md`, and fails the push when they disagree. The sibling
project trailryx has one (`scripts/readme-numbers.sh`); copy the idea, not the
file, since the numbers differ.

Until that exists, invariant 2 is held by attention, and attention is exactly
what failed the three times it has already failed here.

Invariant 4 is partly checkable: assert the default install pulls nothing that
opens a socket. Invariants 3, 5, 7 and 8 are judgement.

## Standing rule

An approved architecture decision is **not finished** until it is two things: a
numbered invariant in this file, and a gate in a script or a test if it can be
checked structurally. Until then it is a document, and documents do not stop
code.

When the user approves a decision, add it here in the same session.

## Escalate, do not push through

Stop and tell the user, then wait, when a task hits any of these:

- Any change to a default, a weight, or a retrieval algorithm, because the
  published numbers move with it (invariant 2).
- Any change to the on-disk schema or the storage format.
- Adding a runtime dependency.
- Cutting a release or publishing to PyPI.

Routine work: tests, docs, new deterministic helpers, refactors that keep the
public API and every published number identical.

## Style

Be critical, not sycophantic. If a proposal contradicts `DESIGN.md`, breaks an
invariant, or has a subtle bug, push back and explain the trade-off honestly.
Brevity over fluff. No emoji in code or commits.

## Conventions

- **No long dashes** anywhere: not in code comments, docs, commit messages, or
  PR bodies. Use a comma, a colon, parentheses, or a short hyphen.
- Nothing paid or metered gets enabled without telling the user first and
  getting agreement. In this repo that specifically includes any benchmark run
  that calls a priced provider.
- Do not delete or revoke keys, tokens, or certificates on your own initiative.
