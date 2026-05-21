"""LlamaIndex adapter for Engram.

Requires: pip install 'engram[llamaindex]'

Provides:
- EngramMemory  — LlamaIndex BaseMemory backed by Engram episodic store
"""

from __future__ import annotations

from typing import Any

try:
    from llama_index.core.llms import (  # type: ignore[import-not-found]
        ChatMessage,
        MessageRole,
    )
    from llama_index.core.memory import BaseMemory  # type: ignore[import-not-found]
    from pydantic import PrivateAttr  # type: ignore[import-not-found]

    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    _LLAMAINDEX_AVAILABLE = False


def _require_llamaindex() -> None:
    if not _LLAMAINDEX_AVAILABLE:
        raise ImportError("llama-index-core not installed. Run: pip install 'engram[llamaindex]'")


if _LLAMAINDEX_AVAILABLE:

    class EngramMemory(BaseMemory):  # type: ignore[misc]
        """LlamaIndex memory buffer backed by Engram episodic store.

        Each :meth:`put` persists the message to Engram so it survives
        session restarts. :meth:`get` with a query string performs semantic
        recall; without a query it returns the most recent *k* messages.

        Args:
            engram_path: Path to the ``.engram`` file (``":memory:"`` for ephemeral).
            k: Number of messages to return from :meth:`get`.

        Example::

            from engram.adapters.llamaindex import EngramMemory

            memory = EngramMemory.from_defaults(engram_path="./agent.engram", k=5)
            memory.put(ChatMessage(role=MessageRole.USER, content="Hello!"))
            msgs = memory.get("Hello")
        """

        engram_path: str = ":memory:"
        k: int = 5

        model_config: Any = {"arbitrary_types_allowed": True}  # type: ignore[misc]  # noqa: RUF012

        _engram: Any = PrivateAttr(default=None)
        _history: Any = PrivateAttr(default_factory=list)

        def model_post_init(self, __context: Any) -> None:
            from engram.core import Engram

            self._engram = Engram(path=self.engram_path)
            self._history = []

            # Hydrate existing messages from the database
            try:
                episodes = self._engram._store.get_episodes_since(since=None, limit=1000)
                role_values = {role.value for role in MessageRole}
                for ep in episodes:
                    role = None
                    for tag in ep.tags:
                        if tag in role_values:
                            role = MessageRole(tag)
                            break
                        # Handle common aliases / fallbacks
                        elif tag == "assistant":
                            role = MessageRole.ASSISTANT
                            break
                        elif tag == "chatbot":
                            role = MessageRole.CHATBOT
                            break
                        elif tag == "user":
                            role = MessageRole.USER
                            break
                        elif tag == "system":
                            role = MessageRole.SYSTEM
                            break
                    if role is not None:
                        self._history.append(ChatMessage(role=role, content=ep.content))
            except Exception:
                pass

        @classmethod
        def from_defaults(cls, **kwargs: Any) -> EngramMemory:
            return cls(**kwargs)

        def get(self, input: str | None = None, **kwargs: Any) -> list[ChatMessage]:  # noqa: A002
            """Return recent messages, optionally re-ranked by semantic similarity."""
            if input:
                results = self._engram.recall(input, k=self.k)
                chat_messages = []
                for r in results:
                    role = MessageRole.USER
                    for tag in r.episode.tags:
                        try:
                            role = MessageRole(tag)
                            break
                        except ValueError:
                            if tag == "assistant" or tag == "chatbot":
                                role = MessageRole.ASSISTANT
                                break
                            elif tag == "user":
                                role = MessageRole.USER
                                break
                            elif tag == "system":
                                role = MessageRole.SYSTEM
                                break
                    chat_messages.append(ChatMessage(role=role, content=r.episode.content))
                return chat_messages
            return list(self._history[-self.k :])

        def get_all(self) -> list[ChatMessage]:
            return list(self._history)

        def put(self, message: ChatMessage) -> None:
            """Persist a message to Engram and append to in-memory history."""
            role_str = (
                str(message.role.value) if hasattr(message.role, "value") else str(message.role)
            )
            self._engram.observe(
                str(message.content),
                tags=[role_str],
            )
            self._history.append(message)

        def set(self, messages: list[ChatMessage]) -> None:
            self._history = list(messages)

        def reset(self) -> None:
            """Clear in-memory history (Engram episodes are retained)."""
            self._history.clear()

else:
    # Stub class that raises on instantiation when llama-index is absent.

    class EngramMemory:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_llamaindex()

        @classmethod
        def from_defaults(cls, **kwargs: Any) -> EngramMemory:
            _require_llamaindex()
            raise RuntimeError("unreachable")
