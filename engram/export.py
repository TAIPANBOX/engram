"""JSON export and import for Engram stores."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from engram.core import Engram


def export_json(mem: Engram, dest: str | Path) -> dict[str, Any]:
    """Export the store to a JSON file and return the document.

    The export contains episodes, facts, entities, and graph edges. It is
    suitable for backup, migration between stores, or offline inspection.
    Access log and reflection run records are not exported (operational data).

    Scoped to *mem*'s ``agent_id``, so an export can never carry more than the
    instance can already read. Episodes and edges carry an ``agent_id`` and are
    filtered to it; both are scoped on every other read path (``get_episode``,
    ``recall``, ``prune_episodes``, ``decay``, ``get_neighbors``), and export
    was the one path that was not. Facts and entities have no ``agent_id``
    column: they are shared across the agents in one file by design (see
    CHANGELOG 2.2.0 and 2.2.1), every fact and entity read path is already
    cross-agent, and so they are exported whole rather than "fixed" into a
    scoping the store never had. An unscoped instance exports everything, which
    is what an operator taking a whole-store dump opens.

    Args:
        mem: Open Engram instance to export from.
        dest: Path to write the JSON file. Parent directory must exist.

    Returns:
        The exported document as a plain dict.
    """
    mem._store.flush_access_log()
    conn = mem._store._conn

    agent_id = mem._agent_id
    scope = "WHERE agent_id = ?" if agent_id is not None else ""
    scope_params: tuple[Any, ...] = (agent_id,) if agent_id is not None else ()

    episodes = [
        dict(row)
        for row in conn.execute(
            "SELECT id, content, timestamp, actors, tags, salience, "
            "emotional_valence, summary_of, importance_score, agent_id "
            f"FROM episodes {scope} ORDER BY timestamp ASC",
            scope_params,
        ).fetchall()
    ]

    facts = [
        dict(row)
        for row in conn.execute(
            "SELECT id, subject, predicate, object, valid_from, valid_to, "
            "recorded_at, superseded_at, superseded_by, confidence, "
            "derived_from, extracted_by FROM facts ORDER BY valid_from ASC"
        ).fetchall()
    ]

    entities = [
        dict(row)
        for row in conn.execute(
            "SELECT id, name, type, aliases, first_seen, last_seen FROM entities ORDER BY name ASC"
        ).fetchall()
    ]

    edges = [
        dict(row)
        for row in conn.execute(
            "SELECT src_id, dst_id, relation, weight, created_at, agent_id "
            f"FROM edges {scope} ORDER BY created_at ASC",
            scope_params,
        ).fetchall()
    ]

    doc: dict[str, Any] = {
        "engram_export_version": 1,
        "exported_at": datetime.now(tz=UTC).isoformat(),
        "counts": {
            "episodes": len(episodes),
            "facts": len(facts),
            "entities": len(entities),
            "edges": len(edges),
        },
        "episodes": episodes,
        "facts": facts,
        "entities": entities,
        "edges": edges,
    }

    Path(dest).write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return doc


def import_json(mem: Engram, src: str | Path, *, merge: bool = False) -> dict[str, int]:
    """Import episodes, facts, entities, and edges from a JSON export file.

    Rows keep the ``agent_id`` recorded in the document rather than being
    re-stamped with *mem*'s own scope, which is what makes a migration between
    stores preserve who wrote what. The consequence is worth knowing: importing
    a document that names another agent writes rows this instance cannot then
    read back through its own scoped recall.

    Args:
        mem: Open Engram instance to import into.
        src: Path to the JSON file produced by :func:`export_json`.
        merge: If ``False`` (default), raises ``ValueError`` when an episode or
            fact id already exists in the target store. If ``True``, skips
            duplicate ids silently (useful for merging two stores).

    Returns:
        Dict with counts of inserted rows per table.
    """
    doc: dict[str, Any] = json.loads(Path(src).read_text(encoding="utf-8"))

    version = doc.get("engram_export_version", 0)
    if version != 1:
        raise ValueError(f"Unsupported export version: {version!r}")

    conn = mem._store._conn
    counts: dict[str, int] = {"episodes": 0, "facts": 0, "entities": 0, "edges": 0}

    on_conflict = "OR IGNORE" if merge else "OR ABORT"

    for ep in doc.get("episodes", []):
        conn.execute(
            f"INSERT {on_conflict} INTO episodes "
            "(id, content, timestamp, actors, tags, salience, emotional_valence, "
            "summary_of, importance_score, agent_id) "
            "VALUES (:id, :content, :timestamp, :actors, :tags, :salience, "
            ":emotional_valence, :summary_of, :importance_score, :agent_id)",
            ep,
        )
        if conn.execute("SELECT changes()").fetchone()[0]:
            # Embeddings aren't carried in the export, so re-embed the content
            # here. Both the vec0 index and the FTS index must be populated or
            # the imported episode is invisible to vector / hybrid recall.
            embedding = mem._embedder.embed(ep["content"]).astype(np.float32).tobytes()
            rowid = conn.execute("SELECT rowid FROM episodes WHERE id = ?", (ep["id"],)).fetchone()[
                0
            ]
            conn.execute(
                "INSERT OR IGNORE INTO vec_episodes(rowid, agent_id, ts, embedding) "
                "VALUES (?, ?, ?, ?)",
                (rowid, ep["agent_id"], ep["timestamp"], embedding),
            )
            conn.execute(
                "INSERT OR IGNORE INTO fts_episodes(rowid, content) VALUES (?, ?)",
                (rowid, ep["content"]),
            )
            counts["episodes"] += 1

    for fact in doc.get("facts", []):
        conn.execute(
            f"INSERT {on_conflict} INTO facts "
            "(id, subject, predicate, object, valid_from, valid_to, recorded_at, "
            "superseded_at, superseded_by, confidence, derived_from, extracted_by) "
            "VALUES (:id, :subject, :predicate, :object, :valid_from, :valid_to, "
            ":recorded_at, :superseded_at, :superseded_by, :confidence, "
            ":derived_from, :extracted_by)",
            fact,
        )
        if conn.execute("SELECT changes()").fetchone()[0]:
            counts["facts"] += 1

    for entity in doc.get("entities", []):
        conn.execute(
            f"INSERT {on_conflict} INTO entities "
            "(id, name, type, aliases, first_seen, last_seen) "
            "VALUES (:id, :name, :type, :aliases, :first_seen, :last_seen)",
            entity,
        )
        if conn.execute("SELECT changes()").fetchone()[0]:
            counts["entities"] += 1

    for edge in doc.get("edges", []):
        # Older exports (v1, pre-edge-scoping) have no agent_id; default it.
        edge.setdefault("agent_id", None)
        conn.execute(
            f"INSERT {on_conflict} INTO edges "
            "(src_id, dst_id, relation, weight, created_at, agent_id) "
            "VALUES (:src_id, :dst_id, :relation, :weight, :created_at, :agent_id)",
            edge,
        )
        if conn.execute("SELECT changes()").fetchone()[0]:
            counts["edges"] += 1

    conn.commit()
    return counts
