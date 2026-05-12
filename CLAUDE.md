# Engram

## What
Engram is an **embeddable cognitive memory layer for AI agents**. Think "SQLite of agent memory" — a single-file, local-first store that models human-like memory (episodic, semantic, procedural) with temporal validity, importance decay, spreading-activation retrieval, and provenance tracking.

Full design: see `DESIGN.md`. **Read it before starting any non-trivial task.**

## Stack
- **Language**: Python 3.11+
- **Storage**: SQLite + `sqlite-vec` extension (single-file DB)
- **Embeddings**: `fastembed` (ONNX, local, no heavy deps)
- **Reflection LLM**: pluggable (Anthropic / OpenAI / Ollama) — never required at write time
- **Test**: pytest
- **Lint/format**: ruff
- **Type check**: mypy `--strict`
- **Package**: pyproject.toml (PEP 621), built with `hatchling`

## Conventions
- **Type hints everywhere.** Public APIs strictly typed. `mypy --strict` must pass.
- **One module = one concern.** No grab-bag files.
- **No global state.** Pass the `Engram` instance explicitly.
- **No network calls at write time.** Only `reflect()` may call an external LLM.
- **Tests are mandatory.** Every new function gets a test in `tests/`. Run `pytest -x` after each change.
- **Commit small.** One logical change per commit. Conventional Commits style (`feat:`, `fix:`, `refactor:`).
- **Docstrings**: Google style for public API. Skip for trivial private helpers.

## Workflow
1. Always enter **Plan Mode** (`Shift+Tab` twice) before any non-trivial change.
2. Reference `DESIGN.md` for architectural decisions — do not reinvent.
3. After implementation: run `ruff check`, `mypy`, `pytest`. Fix all issues before declaring done.
4. If something in `DESIGN.md` is unclear or self-contradictory, **ask before guessing**.

## Style
Be critical, not sycophantic. If I propose something that contradicts the design, breaks an invariant, or has a subtle bug — push back. Explain trade-offs honestly. Brevity over fluff. No emoji in code or commits.

## Commands
- `pytest` — run all tests
- `pytest -x` — stop on first failure
- `ruff check . --fix` — lint + auto-fix
- `ruff format .` — format
- `mypy engram` — type check
- `python -m engram.cli` — CLI entry (when implemented)
