"""Tests for v0.6 integrations: LangChain, LlamaIndex adapters.

MCP server tests moved to tests/test_mcp_server.py as part of the
feature/mcp-server phase: the old ``observe``/``assert_fact``/``timeline``/
``reflect``/``why`` tool surface tested here previously did not match the
architect's spec for the shipped server (``remember``/``recall``/``why``/
``forget``/``stats``, with ``reflect`` deliberately excluded) and was never
wired to a console entry point. See engram/mcp_server.py's module docstring.
"""

from __future__ import annotations

import pytest

# ------------------------------------------------------------------
# LangChain adapter
# ------------------------------------------------------------------


def test_langchain_retriever_returns_documents() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document

    from engram.adapters.langchain import EngramRetriever
    from engram.core import Engram

    mem = Engram()
    mem.observe("Alice is a senior engineer at Acme")
    retriever = EngramRetriever(engram=mem, k=3)
    docs = retriever.invoke("Alice engineer")
    assert isinstance(docs, list)
    assert len(docs) >= 1
    assert isinstance(docs[0], Document)
    assert "Alice" in docs[0].page_content


def test_langchain_retriever_document_metadata() -> None:
    pytest.importorskip("langchain_core")
    from engram.adapters.langchain import EngramRetriever
    from engram.core import Engram

    mem = Engram()
    mem.observe("Bob joined the Berlin office")
    retriever = EngramRetriever(engram=mem, k=1)
    docs = retriever.invoke("Bob Berlin")
    meta = docs[0].metadata
    assert "id" in meta
    assert "score" in meta
    assert "timestamp" in meta


def test_langchain_retriever_empty_store_returns_empty() -> None:
    pytest.importorskip("langchain_core")
    from engram.adapters.langchain import EngramRetriever
    from engram.core import Engram

    mem = Engram()
    retriever = EngramRetriever(engram=mem, k=5)
    docs = retriever.invoke("anything")
    assert docs == []


def test_langchain_chat_history_stores_messages() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage

    from engram.adapters.langchain import EngramChatMessageHistory
    from engram.core import Engram

    mem = Engram()
    history = EngramChatMessageHistory(engram=mem)
    history.add_messages(
        [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!"),
        ]
    )
    msgs = history.messages
    assert len(msgs) == 2
    assert msgs[0].content == "Hello!"
    assert msgs[1].content == "Hi there!"


def test_langchain_chat_history_clear() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.messages import HumanMessage

    from engram.adapters.langchain import EngramChatMessageHistory
    from engram.core import Engram

    history = EngramChatMessageHistory(engram=Engram())
    history.add_messages([HumanMessage(content="test")])
    history.clear()
    assert history.messages == []


def test_langchain_chat_history_persists_to_engram() -> None:
    """Messages added to history are observable in Engram recall."""
    pytest.importorskip("langchain_core")
    from langchain_core.messages import HumanMessage

    from engram.adapters.langchain import EngramChatMessageHistory
    from engram.core import Engram

    mem = Engram()
    history = EngramChatMessageHistory(engram=mem)
    history.add_messages([HumanMessage(content="Ivan lives in Berlin now")])
    results = mem.recall("Ivan Berlin", k=1)
    assert len(results) == 1
    assert "Ivan" in results[0].episode.content


# ------------------------------------------------------------------
# LlamaIndex adapter
# ------------------------------------------------------------------


def test_llamaindex_memory_put_and_get_all() -> None:
    pytest.importorskip("llama_index")
    from llama_index.core.llms import ChatMessage, MessageRole

    from engram.adapters.llamaindex import EngramMemory

    mem = EngramMemory.from_defaults()
    mem.put(ChatMessage(role=MessageRole.USER, content="Tell me about Ivan"))
    mem.put(ChatMessage(role=MessageRole.ASSISTANT, content="Ivan works at Acme"))
    all_msgs = mem.get_all()
    assert len(all_msgs) == 2


def test_llamaindex_memory_get_with_query() -> None:
    pytest.importorskip("llama_index")
    from llama_index.core.llms import ChatMessage, MessageRole

    from engram.adapters.llamaindex import EngramMemory

    mem = EngramMemory.from_defaults(k=3)
    mem.put(ChatMessage(role=MessageRole.USER, content="Ivan moved to Berlin"))
    results = mem.get("Ivan Berlin")
    assert isinstance(results, list)
    assert len(results) >= 1


def test_llamaindex_memory_reset() -> None:
    pytest.importorskip("llama_index")
    from llama_index.core.llms import ChatMessage, MessageRole

    from engram.adapters.llamaindex import EngramMemory

    mem = EngramMemory.from_defaults()
    mem.put(ChatMessage(role=MessageRole.USER, content="hello"))
    mem.reset()
    assert mem.get_all() == []


def test_llamaindex_memory_set() -> None:
    pytest.importorskip("llama_index")
    from llama_index.core.llms import ChatMessage, MessageRole

    from engram.adapters.llamaindex import EngramMemory

    mem = EngramMemory.from_defaults()
    msgs = [
        ChatMessage(role=MessageRole.USER, content="msg1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="msg2"),
    ]
    mem.set(msgs)
    assert len(mem.get_all()) == 2


def test_llamaindex_memory_get_without_query_returns_recent() -> None:
    pytest.importorskip("llama_index")
    from llama_index.core.llms import ChatMessage, MessageRole

    from engram.adapters.llamaindex import EngramMemory

    mem = EngramMemory.from_defaults(k=2)
    for i in range(5):
        mem.put(ChatMessage(role=MessageRole.USER, content=f"message {i}"))
    recent = mem.get()
    assert len(recent) == 2
    assert recent[-1].content == "message 4"


# ------------------------------------------------------------------
# Adapter hydration from a persisted store
# ------------------------------------------------------------------


def test_langchain_chat_history_hydrates_from_persisted_engram(tmp_path) -> None:
    """Reopening an EngramChatMessageHistory on the same file must restore
    the prior conversation, mapping role tags to the correct LangChain
    message subclasses in chronological order."""
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage

    from engram.adapters.langchain import EngramChatMessageHistory
    from engram.core import Engram

    path = str(tmp_path / "history.engram")

    # Session 1: write messages to disk.
    with Engram(path=path) as mem:
        history = EngramChatMessageHistory(engram=mem)
        history.add_messages(
            [
                SystemMessage(content="you are an assistant"),
                HumanMessage(content="hello there"),
                AIMessage(content="hi, how can I help?"),
                ChatMessage(content="custom role text", role="moderator"),
            ]
        )
        # add_messages stores the role on the message via .type, not the
        # tag — but EngramChatMessageHistory observes with tags=[msg.type],
        # which is a known LangChain attribute (human/ai/system/chat).

    # Session 2: fresh process, same file. Hydration must succeed.
    with Engram(path=path) as mem2:
        restored = EngramChatMessageHistory(engram=mem2)
        kinds = [type(m).__name__ for m in restored.messages]
        contents = [m.content for m in restored.messages]

    assert "SystemMessage" in kinds
    assert "HumanMessage" in kinds
    assert "AIMessage" in kinds
    assert "hello there" in contents
    assert "hi, how can I help?" in contents
    # Chronological order preserved.
    assert contents.index("hello there") < contents.index("hi, how can I help?")


def test_langchain_chat_history_skips_episodes_without_chat_tag(tmp_path) -> None:
    """Episodes without a recognised chat role tag must not appear in the
    hydrated history."""
    pytest.importorskip("langchain_core")
    from engram.adapters.langchain import EngramChatMessageHistory
    from engram.core import Engram

    path = str(tmp_path / "mixed.engram")
    with Engram(path=path) as mem:
        mem.observe("non-chat note", tags=["note"])
        mem.observe("conversation turn", tags=["human"])

    with Engram(path=path) as mem2:
        restored = EngramChatMessageHistory(engram=mem2)

    contents = [m.content for m in restored.messages]
    assert contents == ["conversation turn"]


def test_llamaindex_memory_hydrates_from_persisted_engram(tmp_path) -> None:
    """Reopening EngramMemory on the same path must restore prior turns
    with their roles intact, preserving chronological order."""
    pytest.importorskip("llama_index")
    from llama_index.core.llms import ChatMessage, MessageRole

    from engram.adapters.llamaindex import EngramMemory

    path = str(tmp_path / "lh.engram")

    mem = EngramMemory.from_defaults(engram_path=path, k=10)
    mem.put(ChatMessage(role=MessageRole.USER, content="hi"))
    mem.put(ChatMessage(role=MessageRole.ASSISTANT, content="hello, world"))
    # Engram instance keeps the file handle open — close it to flush.
    mem._engram.close()

    restored = EngramMemory.from_defaults(engram_path=path, k=10)
    all_msgs = restored.get_all()
    contents = [m.content for m in all_msgs]
    roles = [m.role for m in all_msgs]
    assert "hi" in contents
    assert "hello, world" in contents
    assert MessageRole.USER in roles
    assert MessageRole.ASSISTANT in roles
