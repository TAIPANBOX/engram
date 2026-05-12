"""Retrieval logic: maps a query string to a ranked list of SearchResults."""

from __future__ import annotations

from datetime import UTC, datetime

from engram.embedder import Embedder
from engram.models import SearchResult
from engram.store import Store


def recall(
    query: str,
    k: int,
    store: Store,
    embedder: Embedder,
    *,
    mode: str = "cosine",
    depth: int = 2,
    decay: float = 0.5,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
    as_of: datetime | None = None,
    agent_id: str | None = None,
) -> list[SearchResult]:
    """Return the top-k episodes most similar to *query*.

    Args:
        query: Natural-language search query.
        k: Maximum number of results.
        store: Active store instance.
        embedder: Embedder for the query.
        mode: ``"cosine"`` (default) or ``"spreading"`` for graph-based recall.
        depth: BFS hops for spreading mode.
        decay: Activation decay per hop for spreading mode.
        alpha: Cosine weight in spreading score.
        beta: Graph activation weight in spreading score.
        gamma: Importance weight in spreading score.
        as_of: If set, only episodes with timestamp <= as_of are considered.
        agent_id: If set, restrict results to this agent. None searches all agents.

    Returns:
        List of :class:`SearchResult` ordered by descending score.
    """
    if mode == "spreading":
        from engram.graph import spreading_recall

        return spreading_recall(
            query,
            k,
            store,
            embedder,
            depth=depth,
            decay=decay,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            as_of=as_of,
            agent_id=agent_id,
        )

    query_vec = embedder.embed(query)
    if as_of is not None:
        hits = store.search_episodes_as_of(query_vec, k, as_of, agent_id=agent_id)
    else:
        hits = store.search_episodes(query_vec, k, agent_id=agent_id)
    now = datetime.now(tz=UTC)
    results: list[SearchResult] = []
    for rank, (ep, score, dist) in enumerate(hits):
        store.log_access(ep.id, now, query, rank)
        results.append(
            SearchResult(
                episode=ep,
                score=score,
                distance=dist,
                importance=ep.importance_score,
            )
        )
    return results
