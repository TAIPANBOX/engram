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

## Releasing (maintainers)

Releases are fully automated via `release.yml` (OIDC trusted publishing — no PyPI tokens):

1. Move the `[Unreleased]` notes in [CHANGELOG.md](CHANGELOG.md) under a new version heading.
2. Bump `version` in `pyproject.toml` (SemVer).
3. `git tag vX.Y.Z && git push --tags` — CI verifies the tag matches `pyproject.toml`
   and publishes `engdbram` to PyPI.
