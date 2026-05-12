"""Working memory — a limited-capacity, LRU scratchpad for active reasoning.

Modelled on Miller's 7±2 law: the scratchpad holds at most *capacity* items
(default 7).  When full, the least-recently-used item is evicted.  Any evicted
item can optionally be flushed to the long-term Engram store so it is not lost.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engram.core import Engram


@dataclass
class WorkingMemoryItem:
    """A single slot in working memory."""

    key: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


class WorkingMemory:
    """Fixed-capacity LRU scratchpad with optional long-term spillover.

    Args:
        capacity: Maximum number of items (default 7, per Miller's law).
        engram: If supplied, evicted items are automatically written to the
            long-term Engram store via :meth:`~engram.core.Engram.observe`.
            Set ``tags=["working_memory_eviction"]`` on the resulting episode.

    Example::

        wm = WorkingMemory(capacity=5, engram=mem)
        wm.set("task", "Summarise the quarterly report")
        wm.set("context", "Report covers Q3 2025 revenue...")
        item = wm.get("task")
        wm.clear()
    """

    def __init__(self, capacity: int = 7, engram: Engram | None = None) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity!r}")
        self._capacity = capacity
        self._engram = engram
        # OrderedDict gives O(1) LRU move-to-end
        self._store: OrderedDict[str, WorkingMemoryItem] = OrderedDict()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def set(self, key: str, content: str, **metadata: Any) -> None:
        """Write or update an item.  Promotes existing key to MRU position.

        If the scratchpad is full and *key* is new, the LRU item is evicted
        (and flushed to long-term memory if an Engram instance was provided).

        Args:
            key: Unique identifier for this slot.
            content: Text content to store.
            **metadata: Arbitrary extra fields attached to the item.
        """
        if key in self._store:
            # Update in-place, move to MRU end
            item = self._store[key]
            item.content = content
            item.metadata = metadata
            item.accessed_at = datetime.now(tz=UTC)
            self._store.move_to_end(key)
            return

        if len(self._store) >= self._capacity:
            # Evict the LRU item (first in OrderedDict)
            _, evicted = self._store.popitem(last=False)
            self._flush_to_longterm(evicted)

        self._store[key] = WorkingMemoryItem(key=key, content=content, metadata=metadata)

    def get(self, key: str) -> WorkingMemoryItem | None:
        """Retrieve an item by key, promoting it to MRU position.

        Returns:
            The :class:`WorkingMemoryItem`, or ``None`` if not present.
        """
        item = self._store.get(key)
        if item is None:
            return None
        item.accessed_at = datetime.now(tz=UTC)
        self._store.move_to_end(key)
        return item

    def peek(self, key: str) -> WorkingMemoryItem | None:
        """Retrieve an item without changing its LRU position."""
        return self._store.get(key)

    def delete(self, key: str) -> bool:
        """Remove a single item.  Returns True if it existed."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def flush(self) -> int:
        """Write all current items to long-term memory and clear the scratchpad.

        When no Engram instance was provided, items are discarded (same as
        :meth:`clear`) and the returned count reflects how many were dropped.

        Returns:
            Number of items flushed (written or discarded).
        """
        count = len(self._store)
        for item in list(self._store.values()):
            self._flush_to_longterm(item)
        self._store.clear()
        return count

    def clear(self) -> None:
        """Discard all items without writing to long-term memory."""
        self._store.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def items(self) -> list[WorkingMemoryItem]:
        """Return all items from LRU to MRU order."""
        return list(self._store.values())

    @property
    def capacity(self) -> int:
        """Maximum number of items this scratchpad can hold."""
        return self._capacity

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __repr__(self) -> str:
        return f"WorkingMemory(capacity={self._capacity}, size={len(self._store)})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_to_longterm(self, item: WorkingMemoryItem) -> None:
        if self._engram is None:
            return
        self._engram.observe(
            item.content,
            tags=["working_memory_eviction"],
        )
