"""Fixed store and fixed queries for the recall golden test.

Not a test module (no ``test_`` prefix). It exists so that the same corpus and
the same queries can be run against two code versions and their output
compared: the before-change capture is taken from origin/main, the after-change
run from the branch, and the ranking and the scores must not move.

Every timestamp is explicit, so ``recall(as_of=...)`` and the time ordering
never depend on the clock. Episode ids are random UUIDs, so results are
identified by content, which is unique in this corpus.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engram import Engram

AGENT_A = "agent://golden.example/planner"
AGENT_B = "agent://golden.example/support"

# (agent, content, actors, tags, timestamp). The first eight belong to agent A,
# the rest to agent B, so one scoped query sees a strict subset of the file.
CORPUS: list[tuple[str, str, list[str], list[str], datetime]] = [
    (
        AGENT_A,
        "Ivan joined Globex as a site reliability engineer",
        ["Ivan"],
        ["hr"],
        datetime(2026, 6, 2, 9, 0, tzinfo=UTC),
    ),
    (
        AGENT_A,
        "Ivan moved the payment service to the new Kubernetes cluster",
        ["Ivan"],
        ["ops"],
        datetime(2026, 6, 19, 14, 30, tzinfo=UTC),
    ),
    (
        AGENT_A,
        "The payment service throttled requests during the Friday incident",
        [],
        ["ops"],
        datetime(2026, 7, 3, 16, 5, tzinfo=UTC),
    ),
    (
        AGENT_A,
        "Sourdough starter needs feeding every twelve hours in summer",
        [],
        ["home"],
        datetime(2026, 7, 8, 7, 45, tzinfo=UTC),
    ),
    (
        AGENT_A,
        "Rolled back the deployment after the database migration stalled",
        [],
        ["ops"],
        datetime(2026, 7, 14, 22, 10, tzinfo=UTC),
    ),
    (
        AGENT_A,
        "Anna prefers video calls on Tuesday mornings",
        ["Anna"],
        ["calendar"],
        datetime(2026, 7, 21, 10, 0, tzinfo=UTC),
    ),
    (
        AGENT_A,
        "Quarterly budget review moved to the second week of September",
        [],
        ["finance"],
        datetime(2026, 8, 4, 11, 20, tzinfo=UTC),
    ),
    (
        AGENT_A,
        "Globex renewed the support contract for another twelve months",
        ["Ivan"],
        ["sales"],
        datetime(2026, 8, 19, 15, 0, tzinfo=UTC),
    ),
    (
        AGENT_B,
        "Customer Maria reported a duplicate charge on her card",
        ["Maria"],
        ["billing"],
        datetime(2026, 7, 1, 9, 30, tzinfo=UTC),
    ),
    (
        AGENT_B,
        "Refund issued to Maria, the duplicate charge came from a retry",
        ["Maria"],
        ["billing"],
        datetime(2026, 7, 2, 12, 0, tzinfo=UTC),
    ),
    (
        AGENT_B,
        "Password reset emails were delayed by the mail relay for an hour",
        [],
        ["support"],
        datetime(2026, 7, 16, 8, 15, tzinfo=UTC),
    ),
    (
        AGENT_B,
        "Escalated the payment incident to the on-call engineer overnight",
        [],
        ["support"],
        datetime(2026, 7, 18, 23, 40, tzinfo=UTC),
    ),
    (
        AGENT_B,
        "Customer asked whether the rollback changed their saved invoices",
        [],
        ["support"],
        datetime(2026, 8, 10, 13, 25, tzinfo=UTC),
    ),
    (
        AGENT_B,
        "Shared the sourdough recipe with the support team for the bake sale",
        [],
        ["social"],
        datetime(2026, 8, 25, 17, 0, tzinfo=UTC),
    ),
]

# mode, query, k, as_of, scoped to agent A (True) or unscoped (False)
QUERIES: list[tuple[str, str, int, datetime | None, bool]] = [
    ("hybrid", "where does Ivan work", 5, None, False),
    ("hybrid", "payment service incident", 5, None, False),
    ("hybrid", "payment service incident", 3, None, True),
    ("hybrid", "sourdough starter feeding", 4, None, False),
    ("cosine", "rollback after the migration", 5, None, False),
    ("cosine", "customer refund for a duplicate charge", 4, None, False),
    ("cosine", "Ivan and the cluster", 3, datetime(2026, 7, 1, tzinfo=UTC), False),
    ("spreading", "payment incident escalation", 4, None, False),
    ("hybrid", "budget review", 5, datetime(2026, 8, 5, tzinfo=UTC), True),
]


def build_store(path: str | Path) -> None:
    """Write the fixed corpus into a store file, one Engram per agent."""
    for agent in (AGENT_A, AGENT_B):
        with Engram(path=path, agent_id=agent) as mem:
            for agent_of_row, content, actors, tags, ts in CORPUS:
                if agent_of_row == agent:
                    mem.observe(content, actors=actors, tags=tags, timestamp=ts)


def run_queries(path: str | Path) -> list[dict[str, Any]]:
    """Run every fixed query against the store and return full-precision results."""
    out: list[dict[str, Any]] = []
    for mode, query, k, as_of, scoped in QUERIES:
        agent = AGENT_A if scoped else None
        with Engram(path=path, agent_id=agent) as mem:
            hits = mem.recall(query, k=k, mode=mode, as_of=as_of)
            out.append(
                {
                    "mode": mode,
                    "query": query,
                    "k": k,
                    "as_of": as_of.isoformat() if as_of else None,
                    "agent": agent,
                    "results": [
                        {
                            "content": hit.episode.content,
                            "score": hit.score,
                            "distance": hit.distance,
                            "importance": hit.importance,
                        }
                        for hit in hits
                    ],
                }
            )
    return out
