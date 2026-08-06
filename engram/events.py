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

``agent_id`` shape: warned about, never refused
-----------------------------------------------
The shared envelope requires ``agent_id`` to match SPEC.md §3.1's
``agent://<trust-domain>/<name>`` grammar (see :func:`is_canonical_agent_id`),
and Engram's own ``agent_id`` is an opaque scoping key that has never had to
be one. So an id like ``"planner"`` produces a store that works and lines that
a consumer validating the envelope rejects.

An id that cannot validate is WARNED about, once per id, counted in
:attr:`EventLog.nonconforming_agent_id`, and written anyway. Refusing to emit
would make the event log silently empty for exactly the caller who needs to
see the problem, and validating in ``Engram.__init__`` would refuse to open
stores that have worked for two releases over a rule that belongs to the wire
rather than to the file. Losing fidelity in an event log a consumer rejects is
recoverable; a caller who cannot open their own store is not.

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

``prev_hash`` chain (SPEC.md §6.5)
------------------------------------
Each events file maintains its own append-only hash chain: every event's
``prev_hash`` is ``"sha256:" + hex(sha256(C))``, where ``C`` is the RFC 8785
(JSON Canonicalization Scheme) serialization of the PREVIOUS event in this
file with its own ``prev_hash`` removed -- see :func:`canonicalize` and
:func:`chain_hash`. The first event in a file carries no ``prev_hash``, and
reopening an existing file resumes the chain from its tail instead of
starting a new one, so one file stays one chain across process restarts.
Resuming is fail-open like the rest of this module: an absent, empty, or
malformed tail starts a fresh chain rather than blocking construction.

This is tamper-EVIDENCE, not tamper-proof: it makes a dropped or edited
line detectable, not impossible for an attacker who can rewrite the whole
file. Verify a file with ``agent-conform -chain <file>``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import rfc8785

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

#: SPEC.md §3.1 grammar for a canonical agent identifier, and the envelope
#: schema's own ``agent_id`` constraints (``agent-event.schema.json``:
#: ``pattern`` plus ``maxLength``). Written here as well as in the schema
#: because this module needs it at runtime and the schema is a vendored copy
#: of somebody else's file; ``test_the_grammar_here_is_the_one_in_the_schema``
#: holds the two equal so they cannot drift.
AGENT_ID_PATTERN = re.compile(r"^agent://[a-z0-9.-]+/[a-z0-9._/-]+$")
AGENT_ID_MAX_LENGTH = 255


def is_canonical_agent_id(agent_id: str) -> bool:
    """Report whether *agent_id* is one the shared envelope schema accepts.

    This is a question about the WIRE, not about the store. Engram scopes
    rows in a local SQLite file by this string and does not care what shape
    it has; a consumer reading the NDJSON event log validates it against
    ``agent-event.schema.json`` and rejects the line if it does not match.
    """
    return len(agent_id) <= AGENT_ID_MAX_LENGTH and AGENT_ID_PATTERN.match(agent_id) is not None


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


#: How far back :func:`EventLog.__post_init__` reads when resuming an
#: existing file's chain (SPEC.md §6.5). One complete event line always
#: fits comfortably inside this window: real envelopes run to a few hundred
#: bytes, and 1 MiB is orders of magnitude beyond any single line this
#: module writes.
_RESUME_WINDOW = 1 << 20


def canonicalize(envelope: dict[str, Any]) -> bytes:
    """Return the RFC 8785 (JCS) canonical serialization of *envelope*.

    SPEC.md §6.5: the chain-hash input is the JCS canonical form of an event
    object with its own ``prev_hash`` field removed. The removal happens on
    a shallow copy -- the caller's dict is never mutated -- and
    canonicalization itself (key sorting, number/string normalization) is
    delegated entirely to ``rfc8785`` (Trail of Bits), never hand-rolled.
    """
    copy = dict(envelope)
    copy.pop("prev_hash", None)
    return rfc8785.dumps(copy)


def chain_hash(envelope: dict[str, Any]) -> str:
    """Return the SPEC.md §6.5 hash of *envelope*.

    ``"sha256:" + hex(sha256(canonicalize(envelope)))`` -- the value the
    NEXT event in a chained NDJSON stream carries as its own ``prev_hash``.
    """
    digest = hashlib.sha256(canonicalize(envelope)).hexdigest()
    return f"sha256:{digest}"


def _tail_chain_hash(path: Path) -> str | None:
    """Resume a chain from *path*'s existing tail (SPEC.md §6.5).

    Reads at most the last :data:`_RESUME_WINDOW` bytes of *path*, keeps the
    last non-blank line, and parses it as one event. Returns that event's
    :func:`chain_hash` so the next :meth:`EventLog.emit` call re-links to
    what is actually on disk -- one file stays one chain across process
    restarts.

    Fail-open, mirroring the rest of this module: a missing or empty file,
    a tail that is not valid JSON or not a JSON object, or any I/O error
    along the way all yield ``None`` (start a fresh chain) rather than
    raising. A malformed tail is exactly the same "start fresh" case as no
    file at all -- nothing here ever blocks construction.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None

    try:
        with path.open("rb") as fh:
            start = max(0, size - _RESUME_WINDOW)
            fh.seek(start)
            tail = fh.read()
    except OSError:
        return None

    lines = tail.split(b"\n")
    if start > 0:
        # A mid-file cut: the first scanned line is likely partial.
        lines = lines[1:]

    last: bytes | None = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped:
            last = stripped
    if last is None:
        return None

    try:
        parsed = json.loads(last)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    try:
        return chain_hash(parsed)
    except rfc8785.CanonicalizationError:
        return None


@dataclass
class EventLog:
    """Appends Agent Passport event envelopes (SPEC.md §6) to an NDJSON file.

    One instance is owned per :class:`~engram.core.Engram` store. The
    destination file is opened in append mode on every :meth:`emit` call
    rather than held open, so a deleted parent directory or revoked
    permission surfaces as a logged warning on the next write rather than a
    crash at construction time.

    Construction also seeds the SPEC.md §6.5 ``prev_hash`` chain from the
    file's existing tail, if any (see :func:`_tail_chain_hash`), so a fresh
    :class:`EventLog` opened over a file another instance already wrote
    resumes the same chain rather than restarting it.

    Args:
        path: Destination NDJSON file. Parent directory must already exist;
            if it does not, writes fail open (see the module docstring).
    """

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    skipped_empty_agent_id: int = field(default=0, init=False)
    #: How many events were written under an ``agent_id`` the envelope schema
    #: will reject (see :func:`is_canonical_agent_id`). Counted for callers
    #: that want the number; what an operator actually sees is the warning
    #: logged the first time each such id is used, not this field.
    nonconforming_agent_id: int = field(default=0, init=False)
    _warned_agent_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _next_hash: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._next_hash = _tail_chain_hash(self.path)

    def _note_agent_id(self, agent_id: str, events: int = 1) -> None:
        """Warn once, count every time, and write the event regardless.

        Refusing to emit was the other candidate and loses on the question
        that decides it: a caller whose id is the wrong shape has a store
        that works and an event log that would then be permanently empty,
        which is the "nobody checked" failure with a second one on top. The
        line is written, the consumer's own validator is what rejects it, and
        this is where somebody is told before that happens.

        Once per distinct id, because a write loop must not turn one
        misconfigured id into a log flood, and the count is what keeps the
        number true after the warning has been said.
        """
        if is_canonical_agent_id(agent_id):
            return
        self.nonconforming_agent_id += events
        if agent_id in self._warned_agent_ids:
            return
        # Bounded: this is normally one id per log, but ``emit`` takes the id
        # per call, so a caller with many of them must not grow this set
        # without limit. Past the cap the count stays true and the warning
        # stops repeating, which is the same trade the cap exists for.
        if len(self._warned_agent_ids) < 64:
            self._warned_agent_ids.add(agent_id)
        logger.warning(
            "engram.events: agent_id %r does not match the Agent Passport grammar "
            "%s (max %d chars), so consumers that validate the shared envelope will "
            "reject these lines. The events are still written. Use an "
            "agent://<trust-domain>/<name> identifier, e.g. "
            "agent://acme.example/planner.",
            agent_id,
            AGENT_ID_PATTERN.pattern,
            AGENT_ID_MAX_LENGTH,
        )

    def _envelope(
        self,
        event_type: str,
        agent_id: str,
        data: dict[str, Any],
        run_id: str | None,
    ) -> dict[str, Any]:
        """Build one SPEC.md §6 envelope, without the chain link.

        Shared by :meth:`emit` and :meth:`emit_many` so a single write and a
        batched one cannot drift into two different envelopes.
        """
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
        return envelope

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

        The event is stamped with the SPEC.md §6.5 ``prev_hash`` chain
        (omitted at a chain head, i.e. the first event in a fresh file) and
        the chain only advances after a successful write, all under
        :attr:`_lock` -- see the module docstring.

        Args:
            event_type: One of the fixed engram event types documented on
                :data:`EVENT_SEVERITY`. Unknown types default to ``"info"``
                severity rather than raising, so a future new type never
                breaks emission.
            agent_id: The instance's agent scope (an opaque string, e.g. an
                Agent Passport ``agent://...`` URI). If ``None`` or empty,
                the event is skipped and counted in
                :attr:`skipped_empty_agent_id` -- Engram never fabricates an
                agent_id to satisfy the envelope's required field. If present
                but not a canonical ``agent://`` identifier, the event is
                still written, and warned about once and counted in
                :attr:`nonconforming_agent_id` (see :meth:`_note_agent_id`).
            data: Free-form event payload, owned by the caller.
            run_id: Optional task-execution correlation id.
        """
        if not agent_id:
            self.skipped_empty_agent_id += 1
            return
        self._note_agent_id(agent_id)

        try:
            envelope = self._envelope(event_type, agent_id, data, run_id)

            with self._lock:
                if self._next_hash:
                    envelope["prev_hash"] = self._next_hash
                line = json.dumps(envelope, separators=(",", ":")) + "\n"
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                self._next_hash = chain_hash(envelope)
        except OSError:
            logger.warning(
                "engram.events: failed to write %r event to %s (event dropped)",
                event_type,
                self.path,
                exc_info=True,
            )

    def emit_many(
        self,
        event_type: str,
        agent_id: str | None,
        data_items: list[dict[str, Any]],
        *,
        run_id: str | None = None,
    ) -> None:
        """Append one event per entry in *data_items*, in a single open.

        Same envelope, same chain and the same fail-open contract as
        :meth:`emit`: this is a loop over :meth:`emit` with the file opened
        once and the lock taken once, not a different kind of event. A batch
        of N memories costs one open and one write rather than N of each,
        which is what lets a bulk path stay a bulk path.

        One line per memory, deliberately, rather than one summary line per
        batch. The envelope carries one ``memory_id`` per event (SPEC.md
        §6.2), so a batched payload would be a second data shape under an
        existing type and every consumer would have to learn it. It also
        keeps each line far inside the :data:`_RESUME_WINDOW` that
        :func:`_tail_chain_hash` reads when resuming a chain, which a single
        line carrying an unbounded id array would not.

        The lines are built before anything is written, so the peak memory is
        the serialized batch (a few hundred bytes per memory, against the
        1.5 KB per embedding the caller is already holding).

        Args:
            event_type: As :meth:`emit`.
            agent_id: As :meth:`emit`. When absent, every item is skipped and
                counted, exactly as one :meth:`emit` call per item would.
            data_items: One payload per event. An empty list writes nothing
                and does not create the file.
            run_id: Optional task-execution correlation id, applied to every
                event in the batch.
        """
        if not agent_id:
            self.skipped_empty_agent_id += len(data_items)
            return
        if not data_items:
            return
        self._note_agent_id(agent_id, events=len(data_items))

        try:
            with self._lock:
                lines: list[str] = []
                next_hash = self._next_hash
                for data in data_items:
                    envelope = self._envelope(event_type, agent_id, data, run_id)
                    if next_hash:
                        envelope["prev_hash"] = next_hash
                    lines.append(json.dumps(envelope, separators=(",", ":")) + "\n")
                    next_hash = chain_hash(envelope)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write("".join(lines))
                self._next_hash = next_hash
        except OSError:
            logger.warning(
                "engram.events: failed to write %d %r events to %s (events dropped)",
                len(data_items),
                event_type,
                self.path,
                exc_info=True,
            )
