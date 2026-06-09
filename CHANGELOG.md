# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `migrate()` now fails loudly with `ValueError` when an existing store is
  opened with an embedder of a different dimension, instead of silently
  corrupting vector search (the vec0 dimension is immutable after creation).
- CI: `security` job running `pip-audit` against the installed environment.

## [2.1.2] - 2026-05-29

### Fixed
- `prune_episodes` is now scoped to the calling agent's `agent_id` in shared
  multi-agent stores, no longer deletes other agents' low-importance episodes,
  cleans the FTS index, and removes orphaned dst edges.
- Hybrid recall (`mode="hybrid"`) now honors `as_of`; previously the parameter
  was silently dropped.
- FTS5 queries no longer crash on bare `*`, `(`, `OR`, `NOT`, `-`, or `"` in
  user input — each token is escaped and wrapped as a phrase.
- Spreading-activation retrieval no longer double-counts the seed cosine score
  (the spurious `beta * cosine` on top of `alpha * cosine`) and now respects
  the Hebbian edge weight, clamped to `[0, 1]`.
- Embedder L2-normalizes all returned vectors so the "cosine" score is
  metric-correct for every fastembed model, not just bge-small/bge-base; the
  dim is now probed at runtime rather than silently defaulting to 384 for
  unknown models.
- Reflection is hardened against prompt injection: episode bodies are wrapped
  in `<observation>` blocks, the system prompt instructs the model to treat
  them as inert data, all extractions run at `temperature=0`, and any
  LLM-derived confidence is capped at `0.95` (also clamped in
  `reflection.reflect()` for stubs that bypass JSON parsing).
- LangChain `EngramChatMessageHistory` and LlamaIndex `EngramMemory` no longer
  silently swallow hydration errors; failures are logged and hydration aborts
  cleanly.
- mypy strict pass works whether or not the optional `sqlcipher3` package is
  installed (CI was previously red).

### Added
- `Engram.timeline(entity, as_of=…)` routes to `get_facts_as_of`, exposing the
  bitemporal fact path through the public API for the first time.
- `AsyncEngram.recall` accepts `k_inner` and `candidate_limit`; `AsyncEngram.timeline`
  accepts `as_of` — async surface is now at parity with sync.
- Tests covering adapter message-history hydration, FTS5 escape edge cases,
  hybrid + `as_of`, prompt-injection confidence clamping, the spreading-
  activation double-count regression, and edge-weight respect.

### Changed
- `AsyncEngram` uses `asyncio.to_thread(...)` instead of the deprecated
  `asyncio.get_event_loop().run_in_executor(None, ...)`.
- README / DESIGN.md / GETTING_STARTED.md / error messages reference the
  `engdbram` PyPI distribution name consistently. Import name is unchanged
  (`from engram import Engram`).

## [2.1.1] - 2026-05-21

### Added
- GitHub Actions CI workflow (lint, typecheck, tests on Python 3.11 / 3.12).
- `DATA_FLOW.md` documenting the read/write paths and on-disk guarantees.
- `Engram.recall()` exposes `k_inner` and `candidate_limit` as tunables.
- LangChain `EngramChatMessageHistory` and LlamaIndex `EngramMemory` rebuild
  their in-memory conversation buffer from persisted episodes on construction.

### Changed
- PyPI distribution name renamed `engram-ai` → `engdbram` (the `engram` name
  is squatted upstream). Import name unchanged.

## [2.1.0] and earlier

See `git log` for the pre-2.1.1 history.
