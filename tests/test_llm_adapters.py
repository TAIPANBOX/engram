"""Tests for all LLM adapters — Gemini, DeepSeek, Qwen, Kimi."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from engram import (
    DeepSeekAdapter,
    GeminiAdapter,
    KimiAdapter,
    OpenAIAdapter,
    QwenAdapter,
)
from engram.llm import LLMAdapter
from engram.models import Episode


def _episode(content: str) -> Episode:
    return Episode(
        id="test-id",
        content=content,
        timestamp=datetime.now(tz=UTC),
    )


EPISODES = [_episode("Alice joined Globex"), _episode("Ivan transferred from Acme")]

FACTS_JSON = json.dumps(
    [
        {"subject": "Alice", "predicate": "works_at", "object": "Globex", "confidence": 0.9},
    ]
)


# ------------------------------------------------------------------
# GeminiAdapter
# ------------------------------------------------------------------


def _make_gemini_response(text: str, prompt_tokens: int = 10, output_tokens: int = 5) -> MagicMock:
    usage = SimpleNamespace(prompt_token_count=prompt_tokens, candidates_token_count=output_tokens)
    return SimpleNamespace(text=text, usage_metadata=usage)


def test_gemini_default_model() -> None:
    adapter = GeminiAdapter()
    assert adapter.model_name == "gemini-2.0-flash"


def test_gemini_custom_model() -> None:
    adapter = GeminiAdapter(model="gemini-1.5-pro")
    assert adapter.model_name == "gemini-1.5-pro"


def test_gemini_satisfies_protocol() -> None:
    adapter = GeminiAdapter()
    assert isinstance(adapter, LLMAdapter)


def test_gemini_import_error_without_sdk() -> None:
    adapter = GeminiAdapter(api_key="fake")
    with (
        patch.dict("sys.modules", {"google": None, "google.genai": None}),
        pytest.raises(ImportError, match="engram\\[gemini\\]"),
    ):
        adapter._get_client()


def test_gemini_extract_facts() -> None:
    adapter = GeminiAdapter(api_key="fake")
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_gemini_response(FACTS_JSON)

    with (
        patch.object(adapter, "_get_client", return_value=mock_client),
        patch("engram.llm.GeminiAdapter.extract_facts", wraps=adapter.extract_facts),
    ):
        # patch the inner import of types
        mock_types = MagicMock()
        mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
        with patch.dict(
            "sys.modules",
            {"google.genai.types": mock_types, "google.genai": MagicMock(types=mock_types)},
        ):
            facts, tokens = adapter.extract_facts(EPISODES)
    assert len(facts) == 1
    assert facts[0]["subject"] == "Alice"
    assert tokens == 15


def test_gemini_summarise() -> None:
    adapter = GeminiAdapter(api_key="fake")
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_gemini_response(
        "Alice joined Globex; Ivan transferred from Acme.", 8, 12
    )

    with patch.object(adapter, "_get_client", return_value=mock_client):
        mock_types = MagicMock()
        mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
        with patch.dict(
            "sys.modules",
            {"google.genai.types": mock_types, "google.genai": MagicMock(types=mock_types)},
        ):
            summary, tokens = adapter.summarise(EPISODES)
    assert "Alice" in summary
    assert tokens == 20


def test_gemini_no_usage_metadata() -> None:
    adapter = GeminiAdapter(api_key="fake")
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        text=FACTS_JSON, usage_metadata=None
    )
    with patch.object(adapter, "_get_client", return_value=mock_client):
        mock_types = MagicMock()
        mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
        with patch.dict(
            "sys.modules",
            {"google.genai.types": mock_types, "google.genai": MagicMock(types=mock_types)},
        ):
            _, tokens = adapter.extract_facts(EPISODES)
    assert tokens == 0


# ------------------------------------------------------------------
# DeepSeekAdapter
# ------------------------------------------------------------------


def test_deepseek_default_model() -> None:
    adapter = DeepSeekAdapter()
    assert adapter.model_name == "deepseek-chat"


def test_deepseek_base_url() -> None:
    adapter = DeepSeekAdapter()
    assert adapter._base_url == "https://api.deepseek.com/v1"


def test_deepseek_custom_model() -> None:
    adapter = DeepSeekAdapter(model="deepseek-reasoner")
    assert adapter.model_name == "deepseek-reasoner"


def test_deepseek_satisfies_protocol() -> None:
    assert isinstance(DeepSeekAdapter(), LLMAdapter)


def test_deepseek_is_openai_subclass() -> None:
    assert isinstance(DeepSeekAdapter(), OpenAIAdapter)


def test_deepseek_import_error_without_sdk() -> None:
    adapter = DeepSeekAdapter(api_key="fake")
    with (
        patch.dict("sys.modules", {"openai": None}),
        pytest.raises(ImportError, match="engram\\[openai\\]"),
    ):
        adapter._get_client()


def _make_openai_response(content: str, tokens: int = 15) -> MagicMock:
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(total_tokens=tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_deepseek_extract_facts() -> None:
    adapter = DeepSeekAdapter(api_key="fake")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(FACTS_JSON)
    with patch.object(adapter, "_get_client", return_value=mock_client):
        facts, tokens = adapter.extract_facts(EPISODES)
    assert len(facts) == 1
    assert tokens == 15


def test_deepseek_summarise() -> None:
    adapter = DeepSeekAdapter(api_key="fake")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(
        "Alice joined Globex; Ivan transferred.", 20
    )
    with patch.object(adapter, "_get_client", return_value=mock_client):
        summary, tokens = adapter.summarise(EPISODES)
    assert "Alice" in summary
    assert tokens == 20


# ------------------------------------------------------------------
# QwenAdapter
# ------------------------------------------------------------------


def test_qwen_default_model() -> None:
    assert QwenAdapter().model_name == "qwen-max"


def test_qwen_base_url() -> None:
    assert QwenAdapter()._base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_qwen_custom_model() -> None:
    assert QwenAdapter(model="qwen-plus").model_name == "qwen-plus"


def test_qwen_satisfies_protocol() -> None:
    assert isinstance(QwenAdapter(), LLMAdapter)


def test_qwen_extract_facts() -> None:
    adapter = QwenAdapter(api_key="fake")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(FACTS_JSON)
    with patch.object(adapter, "_get_client", return_value=mock_client):
        facts, _ = adapter.extract_facts(EPISODES)
    assert len(facts) == 1


def test_qwen_summarise() -> None:
    adapter = QwenAdapter(api_key="fake")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("Summary text.", 10)
    with patch.object(adapter, "_get_client", return_value=mock_client):
        summary, _ = adapter.summarise(EPISODES)
    assert summary == "Summary text."


# ------------------------------------------------------------------
# KimiAdapter
# ------------------------------------------------------------------


def test_kimi_default_model() -> None:
    assert KimiAdapter().model_name == "moonshot-v1-8k"


def test_kimi_base_url() -> None:
    assert KimiAdapter()._base_url == "https://api.moonshot.cn/v1"


def test_kimi_custom_model() -> None:
    assert KimiAdapter(model="moonshot-v1-32k").model_name == "moonshot-v1-32k"


def test_kimi_satisfies_protocol() -> None:
    assert isinstance(KimiAdapter(), LLMAdapter)


def test_kimi_extract_facts() -> None:
    adapter = KimiAdapter(api_key="fake")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(FACTS_JSON)
    with patch.object(adapter, "_get_client", return_value=mock_client):
        facts, _ = adapter.extract_facts(EPISODES)
    assert len(facts) == 1


def test_kimi_summarise() -> None:
    adapter = KimiAdapter(api_key="fake")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("Kimi summary.", 8)
    with patch.object(adapter, "_get_client", return_value=mock_client):
        summary, _ = adapter.summarise(EPISODES)
    assert summary == "Kimi summary."


# ------------------------------------------------------------------
# Common: explicit api_key wires into client kwargs
# ------------------------------------------------------------------


def test_deepseek_api_key_passed_to_client() -> None:
    mock_openai_mod = MagicMock()
    mock_openai_class = MagicMock(return_value=MagicMock())
    mock_openai_mod.OpenAI = mock_openai_class
    adapter = DeepSeekAdapter(api_key="sk-test-123")
    with patch.dict("sys.modules", {"openai": mock_openai_mod}):
        adapter._client = None  # reset cached client
        adapter._get_client()
    call_kwargs = mock_openai_class.call_args.kwargs
    assert call_kwargs.get("api_key") == "sk-test-123"
    assert call_kwargs.get("base_url") == DeepSeekAdapter._BASE_URL


def test_qwen_api_key_passed_to_client() -> None:
    mock_openai_mod = MagicMock()
    mock_openai_class = MagicMock(return_value=MagicMock())
    mock_openai_mod.OpenAI = mock_openai_class
    adapter = QwenAdapter(api_key="sk-qwen-xyz")
    with patch.dict("sys.modules", {"openai": mock_openai_mod}):
        adapter._client = None
        adapter._get_client()
    call_kwargs = mock_openai_class.call_args.kwargs
    assert call_kwargs.get("api_key") == "sk-qwen-xyz"
    assert call_kwargs.get("base_url") == QwenAdapter._BASE_URL


def test_kimi_api_key_passed_to_client() -> None:
    mock_openai_mod = MagicMock()
    mock_openai_class = MagicMock(return_value=MagicMock())
    mock_openai_mod.OpenAI = mock_openai_class
    adapter = KimiAdapter(api_key="moonshot-key")
    with patch.dict("sys.modules", {"openai": mock_openai_mod}):
        adapter._client = None
        adapter._get_client()
    call_kwargs = mock_openai_class.call_args.kwargs
    assert call_kwargs.get("api_key") == "moonshot-key"
    assert call_kwargs.get("base_url") == KimiAdapter._BASE_URL
