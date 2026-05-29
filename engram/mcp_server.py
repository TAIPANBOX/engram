"""MCP server exposing Engram memory operations as tools.

Run with:
    python -m engram.mcp_server [--path PATH]
    ENGRAM_PATH=./agent.engram python -m engram.mcp_server
"""

from __future__ import annotations

import argparse
import os
from typing import Any


def _build_server(mem: Any) -> Any:
    """Create and return a FastMCP server wired to *mem*.

    Separated from main() so the server object can be introspected in tests.
    """
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("MCP SDK not installed. Run: pip install 'engdbram[mcp]'") from exc

    mcp: Any = FastMCP("engram")

    @mcp.tool()
    def observe(
        content: str,
        actors: list[str] | None = None,
        tags: list[str] | None = None,
        salience: float = 0.5,
        emotional_valence: float = 0.0,
    ) -> dict[str, str]:
        """Record a new episodic memory."""
        episode_id: str = mem.observe(
            content,
            actors=actors,
            tags=tags,
            salience=salience,
            emotional_valence=emotional_valence,
        )
        return {"id": episode_id}

    @mcp.tool()
    def recall(
        query: str,
        k: int = 5,
        mode: str = "cosine",
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k episodes most similar to the query."""
        results = mem.recall(query, k=k, mode=mode)
        return [
            {
                "id": r.episode.id,
                "content": r.episode.content,
                "score": r.score,
                "timestamp": r.episode.timestamp.isoformat(),
            }
            for r in results
        ]

    @mcp.tool()
    def assert_fact(
        subject: str,
        predicate: str,
        object: str,  # noqa: A002
        confidence: float = 1.0,
        source: str | None = None,
    ) -> dict[str, str]:
        """Record a semantic fact triple (subject, predicate, object)."""
        fact_id: str = mem.assert_fact(
            subject,
            predicate,
            object,
            confidence=confidence,
            source=source,
        )
        return {"id": fact_id}

    @mcp.tool()
    def timeline(entity: str) -> list[dict[str, Any]]:
        """Return the full fact history for an entity in chronological order."""
        facts = mem.timeline(entity)
        return [
            {
                "id": f.id,
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "valid_from": f.valid_from.isoformat(),
                "valid_to": f.valid_to.isoformat() if f.valid_to else None,
                "confidence": f.confidence,
            }
            for f in facts
        ]

    @mcp.tool()
    def why(fact_id: str) -> dict[str, Any]:
        """Return provenance information for a fact."""
        result: dict[str, Any] = mem.why(fact_id)
        return result

    @mcp.tool()
    def reflect() -> dict[str, int]:
        """Run the reflection loop: extract facts, decay importance, prune."""
        run = mem.reflect()
        return {
            "episodes_processed": run.episodes_processed,
            "facts_extracted": run.facts_extracted,
            "contradictions_resolved": run.contradictions_resolved,
        }

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Engram MCP server")
    parser.add_argument(
        "--path",
        default=os.environ.get("ENGRAM_PATH", ":memory:"),
        help="Path to .engram file (default: :memory: or $ENGRAM_PATH)",
    )
    args = parser.parse_args()

    from engram.core import Engram

    mem = Engram(path=args.path)
    server = _build_server(mem)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
