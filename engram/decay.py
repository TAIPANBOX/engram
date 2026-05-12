"""Background decay job: recomputes importance scores for all episodes."""

from __future__ import annotations

from datetime import UTC, datetime

from engram.importance import DecayConfig, compute_importance
from engram.store import Store


def run_decay(
    store: Store,
    cfg: DecayConfig,
    now: datetime | None = None,
) -> int:
    """Recompute and persist importance scores for every episode.

    Uses two bulk SQL operations (one GROUP BY fetch + one executemany update)
    instead of N individual round-trips, so performance scales O(1) with episode
    count rather than O(N).

    Args:
        store: Active store instance.
        cfg: Decay hyper-parameters.
        now: Reference time (defaults to current UTC time).

    Returns:
        Number of episodes updated.
    """
    if now is None:
        now = datetime.now(tz=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    episodes = store.get_episodes_for_decay()
    if not episodes:
        return 0

    # Single query: access stats for all episodes at once.
    all_stats = store.get_all_access_stats()

    scores: dict[str, float] = {}
    for ep_id, salience, emotional_valence, timestamp in episodes:
        access_count, last_access = all_stats.get(ep_id, (0, None))
        baseline = last_access if last_access is not None else timestamp
        if baseline.tzinfo is None:
            baseline = baseline.replace(tzinfo=UTC)
        scores[ep_id] = compute_importance(
            salience, emotional_valence, baseline, access_count, now, cfg
        )

    # Single executemany + single commit for all updates.
    store.batch_update_importance(scores)
    return len(episodes)
