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
./scripts/readme-numbers.sh
./scripts/local-first.sh
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
   change**, not a follow-up. *(gate: `scripts/readme-numbers.sh`)*
3. **Recall is measured on the full public benchmark, not on a fixture we
   wrote.** A benchmark we author ourselves measures agreement with our own
   assumptions. Where a metric is knowingly an upper or lower bound, say so
   next to it rather than rounding the caveat away. *(not enforced)*
4. **Local-first is a constraint, not a preference.** A single file you can
   copy, with no service to run. A dependency that needs a running server
   breaks the one sentence that describes this project. The embedding model is
   the one exception and it is bounded: `fastembed` fetches it once, on first
   use, and never again. After that a full write-and-recall cycle opens no
   socket, and the size of that one fetch is published.
   *(gate: `scripts/local-first.sh`)*
5. **No global state.** Pass the `Engram` instance explicitly. Two stores in
   one process must not be able to see each other.
   *(test: `test_pool_isolates_different_agents`,
   `test_recall_does_not_see_other_agents_episodes`,
   `test_episode_count_scoped_per_agent`, and the rest of
   `tests/test_multiagent.py`, which is the observable form of this claim)*
6. **Public APIs are strictly typed and `mypy --strict` passes.**
   *(gate: `mypy engram` in CI and in the gate list above)*
7. **One module, one concern.** No grab-bag files. *(not enforced)*
8. **Every new function gets a test.** A process rule, not a property of the
   code: nothing can tell a function that should have had a test from one that
   should not. *(not enforced, and not enforceable)*

## Decisions that have no gate yet

This list is debt, and it is here to stay visible rather than to be tidy.

**Held by this file alone: invariants 3 and 7.**

A correction: invariant 5 was listed here and is in fact covered by the whole
of `tests/test_multiagent.py`, which is the observable form of "two stores in
one process must not see each other". The claim was made by reading the code
and never opening the suite. Set a marker from evidence, both ways: before
writing `(not enforced)`, grep the suite; before writing `(test: ...)`, open
the test and check it asserts what the invariant claims.

Invariant 8, "every new function gets a test", is a process rule rather than a
property of the code, so it has no marker of its own and is enforced by review.

Invariant 2 now has `scripts/readme-numbers.sh`, which recomputes the recall
table from the committed per-question records, sums the corpus size from the
same file, takes the test count from the suite, and requires the version to
agree across `pyproject.toml`, `engram/__init__.py` and the README badge.

It was written against the unfixed repository and found two real defects on the
first run: the README claimed 424 tests where the suite collects 466, and
`engram/__init__.py` reported `2.2.1` while `pyproject.toml` declared `2.4.1`,
so an installed package was lying about its own version two releases running.

It also reproduces the failure that 2.4.1 exists to fix. Change the declared
default weights in the README without touching the table and it names all four
numbers that moved. That is the exact bug nobody could catch from the table
alone.

**One thing it got wrong, worth keeping.** The test count depends on the
environment: `tests/test_encryption.py` skips at module level without
`sqlcipher3`, so its 7 tests are not collected at all. The first version
compared a measured environment to a single fixed number and failed in CI on
the very run that first wired it into a workflow, in the one job that installs
the encryption extra. The README now states both counts and the check asks which
environment it is in. Anything that measures the environment must either be
environment-independent or know what it is running on.

What it does NOT check: whether the committed records are themselves a true
measurement. It checks that the published numbers agree with the evidence in
the repo, not that the evidence is right. Re-running the benchmark is a
six-hour pass and stays a deliberate act.

Invariant 4 now has `scripts/local-first.sh`, and writing it changed the
invariant. The old wording said an embedding path requiring a network round
trip breaks the project. Run with a cold cache and the network blocked,
`observe()` fails: `fastembed` fetches the ONNX model on first use. So the
invariant as written was violated by the default install, on every fresh
machine, and had been since the embedder was chosen. The README already said so
plainly in its Requirements line; the invariant had simply never caught up with
it. Rather than pretend, invariant 4 now states the promise that is actually
kept: one fetch, then never again.

**It found a second, user-facing defect on its first run.** The README said the
first-use download is ~23 MB. It is 64.1 MB, effectively all of it
`model_optimized.onnx`. That sentence is the first thing a stranger reads about
what installing costs them, and it was understating by nearly three times.

The check holds three things: the default dependency list is an allow-list of
three local packages, so a new runtime dependency has to be argued for rather
than merely fail to match a denylist somebody wrote in 2026; a full
observe-and-recall cycle opens zero sockets once the model is present, verified
by making socket creation raise rather than by watching traffic, so a call that
would have connected fails loudly instead of passing on a machine with no
route; and the published download size matches the bytes on disk.

It runs in CI rather than in a hook, deliberately. Parts two and three need the
model, and on a cold machine that is a 64 MB fetch. CI is the machine that can
afford it, and it says out loud when it is downloading rather than appearing to
do nothing for a minute.

Two of the five break tests initially reported nothing, and both times the
mutation had not landed: `dependencies = [` is followed by comment lines, and
`def observe(` has a multi-line signature. A mutation that did not apply looks
exactly like a check that passed. Verify the break is in the file before
believing the verdict.

Invariants 3, 7 and 8 are judgement.

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

## Known pitfalls

- **`core.fileMode` is `false` in this repo, so git does not record an
  executable bit.** `chmod +x` succeeds on disk, git ignores it, and a new
  script lands as `100644`. Anyone who clones then gets permission denied,
  which for a gate means it silently does not run. Add executables with
  `git update-index --chmod=+x <path>`. This bit
  `scripts/no-network-at-write.sh` on the commit that introduced it.

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
