"""Retrieval logic: maps a query string to a ranked list of SearchResults."""

from __future__ import annotations

from datetime import UTC, datetime

from engram.embedder import Embedder
from engram.models import SearchResult
from engram.store import Store


def recall(query: str, k: int, store: Store, embedder: Embedder) -> list[SearchResult]:
    """Return the top-k episodes most similar to *query*.

    Access events are logged transparently; importance scores are reflected
    in the returned SearchResult objects.
    """
    query_vec = embedder.embed(query)
    hits = store.search_episodes(query_vec, k)
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
