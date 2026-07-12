# Security Policy

Engram stores an agent's episodic, semantic, and procedural memory, so its
own trust boundaries matter: a corrupted or poisoned memory store is a
credible path to steering an agent's future behavior. This document covers
how to report a vulnerability.

## Reporting a vulnerability

Please report security issues privately, not in public issues or PRs:

- Open a **GitHub private security advisory**:
  <https://github.com/TAIPANBOX/engram/security/advisories/new>

Include the affected version/commit, a description, and a minimal reproduction.
We aim to acknowledge within a few days and to fix high-severity issues before
any public disclosure. There is no bug-bounty program; we credit reporters in
the advisory unless you prefer otherwise.

## Supported versions

Engram is pre-1.0; only `main` is supported. Fixes land on `main` and are not
backported.

## Verifying a build

Every change must pass the full gate before merge: `pytest`, `ruff check .`,
`ruff format --check .`, and `mypy engram` (strict). See
[CONTRIBUTING.md](CONTRIBUTING.md).
