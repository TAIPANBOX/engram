"""LangChain adapters for Engram.

Requires: pip install 'engram[langchain]'

Provides:
- EngramRetriever  — plugs Engram into LangChain retrieval pipelines
- EngramChatMessageHistory  — persists chat turns to Engram episodic memory
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

try:
    from langchain_core.callbacks.manager import (  # type: ignore[import-not-found]
        CallbackManagerForRetrieverRun,
    )
    from langchain_core.chat_history import (  # type: ignore[import-not-found]
        BaseChatMessageHistory,
    )
    from langchain_core.documents import Document  # type: ignore[import-not-found]
    from langchain_core.messages import BaseMessage  # type: ignore[import-not-found]
    from langchain_core.retrievers import BaseRetriever  # type: ignore[import-not-found]
    from pydantic import ConfigDict  # type: ignore[import-not-found]

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

if TYPE_CHECKING:
    pass


def _require_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError("langchain-core not installed. Run: pip install 'engram[langchain]'")


if _LANGCHAIN_AVAILABLE:

    class EngramRetriever(BaseRetriever):  # type: ignore[misc]
        """LangChain retriever backed by Engram vector recall.

        Args:
            engram: An :class:`~engram.core.Engram` instance.
            k: Number of documents to retrieve.
            mode: Retrieval mode — ``"cosine"`` or ``"spreading"``.

        Example::

            from engram import Engram
            from engram.adapters.langchain import EngramRetriever

            mem = Engram(path="./agent.engram")
            retriever = EngramRetriever(engram=mem, k=5)
            docs = retriever.invoke("where does Ivan work?")
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        engram: Any
        k: int = 5
        mode: str = "cosine"

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
        ) -> list[Document]:
            results = self.engram.recall(query, k=self.k, mode=self.mode)
            return [
                Document(
                    page_content=r.episode.content,
                    metadata={
                        "id": r.episode.id,
                        "score": r.score,
                        "timestamp": r.episode.timestamp.isoformat(),
                        "importance": r.importance,
                    },
                )
                for r in results
            ]

    class EngramChatMessageHistory(BaseChatMessageHistory):  # type: ignore[misc]
        """LangChain chat message history backed by Engram episodic memory.

        Persists every message to Engram via :meth:`~engram.core.Engram.observe`
        so they are retrievable across sessions. The current-session ordering is
        maintained in an in-memory list.

        Args:
            engram: An :class:`~engram.core.Engram` instance.

        Example::

            from engram import Engram
            from engram.adapters.langchain import EngramChatMessageHistory

            mem = Engram(path="./agent.engram")
            history = EngramChatMessageHistory(engram=mem)
            history.add_user_message("Hello!")
            history.add_ai_message("Hi there!")
            print(history.messages)
        """

        def __init__(self, engram: Any) -> None:
            self._engram = engram
            self._messages: list[BaseMessage] = []

        @property
        def messages(self) -> list[BaseMessage]:
            return list(self._messages)

        def add_messages(self, messages: Sequence[BaseMessage]) -> None:
            for msg in messages:
                role = getattr(msg, "type", "unknown")
                self._engram.observe(
                    str(msg.content),
                    tags=[role],
                )
                self._messages.append(msg)

        def clear(self) -> None:
            """Clear the in-memory session history.

            Note: episodes already persisted in Engram are not deleted —
            Engram does not support deletion in v0.6.
            """
            self._messages.clear()

else:
    # Provide stub classes that raise on instantiation when langchain is absent.

    class EngramRetriever:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_langchain()

    class EngramChatMessageHistory:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_langchain()
