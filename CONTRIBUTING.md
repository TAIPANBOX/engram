# Contributing to Engram

Thank you for your interest in contributing.

## Before you start

- Open an issue first for non-trivial changes — discuss the approach before writing code.
- Read [DESIGN.md](DESIGN.md) before touching core modules. It explains the cognitive model and architectural invariants.

## Setup

```bash
git clone https://github.com/taipanbox/engram
cd engram
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

## Development loop

```bash
pytest -x           # stop on first failure
ruff check . --fix  # lint + auto-fix
ruff format .       # format
mypy engram         # type check (strict)
```

All four must be clean before submitting a PR.

## Conventions

- **Conventional Commits:** `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- **One PR = one logical change.** Don't bundle refactors with features.
- **Type hints everywhere.** `mypy --strict` must pass.
- **Tests are mandatory.** Every new function needs a test in `tests/`.
- **No global state.** Pass the `Engram` instance explicitly.
- **No network calls at write time.** Only `reflect()` may call an external LLM.
- **Docstrings:** Google style for public API. Skip trivial private helpers.

## What we're looking for

See [Roadmap in README.md](README.md#roadmap) for planned features. Good first contributions:

- Additional test coverage
- Documentation improvements
- Benchmark comparisons against other memory libraries
- Bug fixes with a minimal reproduction

## What we're not looking for (without prior discussion)

- Breaking API changes
- New mandatory dependencies
- Features that require a running server
- LLM calls at write time
