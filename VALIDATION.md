# Live infrastructure validation

Engram's Anthropic adapter was run against a real Claude model on real Linux infrastructure before any
public launch - the first time its reflection pipeline had ever processed live model output rather than
fixtures or a stub. Infrastructure was disposable (ephemeral Hetzner VPS boxes, torn down after each run)
and reachable only over `127.0.0.1` / SSH tunnel.

## Reflection against real Claude output

Three separate runs, on three different box topologies, all with the same result: **zero bugs, zero
contradictions.**

| Run | Episodes in | Facts out | Confidence | Contradictions |
|---|---|---|---|---|
| First run (3-node loopback cluster) | 5 | 10 | scored, full `derived_from` provenance | 0 |
| Cross-machine (4 separate hosts) | 5 | 10 | identical to the first run | 0 |
| Enriched multi-agent campaign | 8 | 17 | 0.80-0.95 range | 0 |

Every extracted fact carries a scored confidence and a full provenance chain back to the observations it
came from - the `why()` differentiator, exercised here on real-model output for the first time rather
than a scripted response.

## What this proves

- The reflection prompt and parsing hold up against real Claude variance (formatting, phrasing,
  occasional verbosity) rather than only the deterministic stub output used in CI.
- Confidence scoring is stable across three independent runs on different infrastructure.
- `why()` provenance - tracing a belief back to the exact episodic observations it was derived from -
  works end to end on real-model output, not just in unit tests.
- No contradiction-detection false negatives across any of the three runs.

## Method

Disposable Hetzner VPS boxes (deleted after each run); code delivered as a `git archive` tarball (no
secrets, no `.git`, no token); the Anthropic key was written to a root-only file, never logged, and
revoked after use; the service bound to `127.0.0.1` only. Nothing from these runs was ever exposed
publicly, and no infrastructure or secret from the campaign persists today.
