# Engram

> Embeddable cognitive memory layer for AI agents.

**Status:** pre-alpha, in active development.

Engram is a single-file, local-first memory store for AI agents. It models human-like memory — episodic, semantic, and procedural — with temporal validity, importance decay, spreading-activation retrieval, and provenance tracking.

Think of it as the **"SQLite of agent memory"**: zero-config, embeddable, opinionated.

## Why

Existing solutions are either too primitive (vector DBs match cosine similarity but know nothing about time, importance, or context) or too heavy (server-based platforms with their own runtime). Engram fills the gap: a lightweight library that gives any agent a cognitively-grounded long-term memory.

See [DESIGN.md](./DESIGN.md) for the full architecture and rationale.

## Quick start

> Not implemented yet. This is the target API.

```python
from engram import Engram

mem = Engram(path="./agent.engram")

mem.observe("Ivan said he moved to Globex", actors=["Ivan"])
results = mem.recall("where does Ivan work?")
```

## Documentation

- [DESIGN.md](./DESIGN.md) — full design document
- [GETTING_STARTED.md](./GETTING_STARTED.md) — build instructions for contributors
- [CLAUDE.md](./CLAUDE.md) — conventions for AI-assisted development

## License

MIT
