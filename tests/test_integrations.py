"""Tests for v0.6 integrations: MCP server, LangChain, LlamaIndex adapters."""

from __future__ import annotations

import pytest

# ------------------------------------------------------------------
# MCP server
# ------------------------------------------------------------------


def test_mcp_server_build_registers_six_tools() -> None:
    """_build_server returns a FastMCP instance with 6 registered tools."""
    mcp_mod = pytest.importorskip("mcp")  # noqa: F841
    from engram.core import Engram
    from engram.mcp_server import _build_server

    mem = Engram()
    server = _build_server(mem)
    # FastMCP exposes registered tools via _tool_manager
    tools = server._tool_manager.list_tools()
    names = {t.name for t in tools}
    assert names == {"observe", "recall", "assert_fact", "timeline", "why", "reflect"}


def test_mcp_observe_tool_returns_id() -> None:
    pytest.importorskip("mcp")
    from engram.core import Engram
    from engram.mcp_server import _build_server

    mem = Engram()
    server = _build_server(mem)
    tool_fn = server._tool_manager._tools["observe"].fn
    result = tool_fn("Alice moved to Berlin")
    assert "id" in result
    assert isinstance(result["id"], str)


def test_mcp_recall_tool_returns_list() -> None:
    pytest.importorskip("mcp")
    from engram.core import Engram
    from engram.mcp_server import _build_server

    mem = Engram()
    mem.observe("Ivan works at Globex")
    server = _build_server(mem)
    tool_fn = server._tool_manager._tools["recall"].fn
    results = tool_fn("Ivan", k=3)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert "content" in results[0]
    assert "score" in results[0]


def test_mcp_assert_fact_tool() -> None:
    pytest.importorskip("mcp")
    from engram.core import Engram
    from engram.mcp_server import _build_server

    mem = Engram()
    server = _build_server(mem)
    tool_fn = server._tool_manager._tools["assert_fact"].fn
    result = tool_fn("Ivan", "works_at", "Globex", confidence=0.9)
    assert "id" in result


def test_mcp_timeline_tool() -> None:
    pytest.importorskip("mcp")
    from engram.core import Engram
    from engram.mcp_server import _build_server

    mem = Engram()
    mem.assert_fact("Ivan", "works_at", "Acme")
    server = _build_server(mem)
    tool_fn = server._tool_manager._tools["timeline"].fn
    result = tool_fn("Ivan")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["predicate"] == "works_at"


def test_mcp_reflect_tool() -> None:
    pytest.importorskip("mcp")
    from engram.core import Engram
    from engram.mcp_server import _build_server

    mem = Engram()
    mem.observe("some event")
    server = _build_server(mem)
    tool_fn = server._tool_manager._tools["reflect"].fn
    result = tool_fn()
    assert "episodes_processed" in result
    assert "facts_extracted" in result
    assert "contradictions_resolved" in result


def test_mcp_why_tool_raises_on_missing_fact() -> None:
    pytest.importorskip("mcp")
    from engram.core import Engram
    from engram.mcp_server import _build_server

    mem = Engram()
    server = _build_server(mem)
    tool_fn = server._tool_manager._tools["why"].fn
    with pytest.raises(KeyError):
        tool_fn("nonexistent-id")


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
