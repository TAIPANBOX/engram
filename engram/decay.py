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

    Args:
        store: Active store instance.
        cfg: Decay hyper-parameters.
        now: Reference time (defaults to current UTC time).

    Returns:
        Number of episodes updated.
    """
    if now is None:
        now = datetime.now(tz=UTC)

    episodes = store.get_episodes_for_decay()
    for ep_id, salience, emotional_valence, timestamp in episodes:
        access_count, last_access = store.get_access_stats(ep_id)
        # Use creation timestamp as baseline if the episode was never accessed.
        baseline = last_access if last_access is not None else timestamp
        # Ensure both datetimes are timezone-aware for subtraction.
        if baseline.tzinfo is None:
            baseline = baseline.replace(tzinfo=UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        score = compute_importance(salience, emotional_valence, baseline, access_count, now, cfg)
        store.update_importance(ep_id, score)

    return len(episodes)
