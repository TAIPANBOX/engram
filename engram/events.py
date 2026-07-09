"""Opt-in NDJSON exporter for Agent Passport events.

See ``agent-passport/SPEC.md`` §6 and ``schemas/agent-event.schema.json`` in
the sibling ``TAIPANBOX/agent-passport`` repo for the wire format this module
implements. Engram is the ``source: "engram"`` emitter in that shared
envelope: every event written here lets the surrounding governance stack
(TokenFuse / Idryx / Qryx) observe what Engram remembered, forgot, or
flagged, without any of those products depending on Engram's internals.

Why this does not violate "no network calls at write time"
------------------------------------------------------------
:class:`EventLog.emit` performs a local filesystem append: open, write,
close. No socket is opened, no DNS lookup happens, no external service is
contacted, and no result is awaited from anywhere off-box. The "no network
calls at write time" invariant exists so Engram stays usable fully offline
and so write latency stays bounded by local disk I/O rather than a remote
service's availability; a local NDJSON append preserves both properties
exactly as writing to the SQLite file itself does. If a future backend wants
to ship these events over the network, that belongs in a separate consumer
process that tails the file -- never in this module.

Opt-in only
-----------
:class:`EventLog` is only ever constructed when the caller asks for it,
either via ``Engram(events_path=...)`` or the ``ENGRAM_EVENTS_PATH``
environment variable (see :func:`resolve_events_path`). When neither is set,
``Engram._events`` stays ``None`` and every call site pays exactly one
``is None`` check -- no file handle, no thread, no allocation.

Fail-open
---------
Any I/O error raised while appending is caught inside :meth:`EventLog.emit`,
logged as a warning, and swallowed. Losing an event is acceptable; losing or
delaying the memory operation the event describes is not. ``emit`` therefore
never raises.

Event types and severities (fixed mapping, per SPEC.md §6.2 "engram" row)
---------------------------------------------------------------------------
=====================  ========  ==============================================
type                   severity  data
=====================  ========  ==============================================
memory_written         info      memory_id, kind ("episodic" | "semantic")
memory_forgotten       info      memory_id, kind ("episodic" | "semantic")
reflection_run         info      cheap summary already computed by reflect()
contradiction_found    medium    memory_id, conflicting_memory_id
=====================  ========  ==============================================

No ``prev_hash``
-----------------
Engram keeps no hash chain over its own writes, so the optional
``prev_hash`` field (present in TokenFuse's audit trail) is always omitted
rather than fabricated.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

#: Schema identifier for the envelope this module emits. See SPEC.md §6.
SCHEMA = "taipanbox.dev/agent-event/v0.1"

#: This module's fixed ``source`` value in the shared envelope.
SOURCE = "engram"

Severity = Literal["info", "low", "medium", "high", "critical"]

#: Fixed event-type -> severity mapping. See the module docstring table.
#: Not user-configurable: severities are a taxonomy decision, not per-call data.
EVENT_SEVERITY: dict[str, Severity] = {
    "memory_written": "info",
    "memory_forgotten": "info",
    "reflection_run": "info",
    "contradiction_found": "medium",
}

#: Environment variable fallback for ``Engram(events_path=...)``.
ENV_EVENTS_PATH = "ENGRAM_EVENTS_PATH"


def resolve_events_path(explicit: str | Path | None) -> Path | None:
    """Resolve the effective events file path.

    Args:
        explicit: The ``events_path`` argument passed to ``Engram(...)``.

    Returns:
        ``Path(explicit)`` if given; otherwise ``Path(ENGRAM_EVENTS_PATH)``
        if that environment variable is set and non-empty; otherwise
        ``None`` (events are fully disabled).
    """
    if explicit is not None:
        return Path(explicit)
    env_value = os.environ.get(ENV_EVENTS_PATH)
    if env_value:
        return Path(env_value)
    return None


def _now_rfc3339() -> str:
    """Return the current UTC time as RFC 3339 with a literal ``Z`` suffix."""
    return datetime.now(tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class EventLog:
    """Appends Agent Passport event envelopes (SPEC.md §6) to an NDJSON file.

    One instance is owned per :class:`~engram.core.Engram` store. The
    destination file is opened in append mode on every :meth:`emit` call
    rather than held open, so a deleted parent directory or revoked
    permission surfaces as a logged warning on the next write rather than a
    crash at construction time.

    Args:
        path: Destination NDJSON file. Parent directory must already exist;
            if it does not, writes fail open (see the module docstring).
    """

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    skipped_empty_agent_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def emit(
        self,
        event_type: str,
        agent_id: str | None,
        data: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> None:
        """Append one event to the NDJSON file.

        Never raises: any failure to resolve or write the event is caught,
        logged as a warning, and swallowed so the caller's memory operation
        always completes.

        Args:
            event_type: One of the fixed engram event types documented on
                :data:`EVENT_SEVERITY`. Unknown types default to ``"info"``
                severity rather than raising, so a future new type never
                breaks emission.
            agent_id: The instance's agent scope (an opaque string, e.g. an
                Agent Passport ``agent://...`` URI). If ``None`` or empty,
                the event is skipped and counted in
                :attr:`skipped_empty_agent_id` -- Engram never fabricates an
                agent_id to satisfy the envelope's required field.
            data: Free-form event payload, owned by the caller.
            run_id: Optional task-execution correlation id.
        """
        if not agent_id:
            self.skipped_empty_agent_id += 1
            return

        try:
            envelope: dict[str, Any] = {
                "schema": SCHEMA,
                "ts": _now_rfc3339(),
                "source": SOURCE,
                "type": event_type,
                "severity": EVENT_SEVERITY.get(event_type, "info"),
                "agent_id": agent_id,
                "data": data,
            }
            if run_id is not None:
                envelope["run_id"] = run_id

            line = json.dumps(envelope, separators=(",", ":")) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            logger.warning(
                "engram.events: failed to write %r event to %s (event dropped)",
                event_type,
                self.path,
                exc_info=True,
            )
