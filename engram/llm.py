"""Pluggable LLM adapters for fact extraction during reflection."""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any, Protocol, runtime_checkable

from engram.models import Episode

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = (
    "You are a semantic fact extractor for an AI memory system. "
    "Given a list of episodic observations, extract atomic facts as "
    "(subject, predicate, object) triples. "
    "subject and object should be concise noun phrases. "
    "predicate should be a short verb phrase (e.g. works_at, lives_in, reported_by). "
    "Respond ONLY with a valid JSON array — no prose, no markdown. "
    'Example: [{"subject": "Ivan", "predicate": "works_at", "object": "Globex", "confidence": 0.9}]'
)

SUMMARISATION_SYSTEM_PROMPT = (
    "You are a memory compressor for an AI agent. "
    "Given a list of episodic observations, write a single concise summary "
    "that preserves all key facts, actors, and events. "
    "Respond ONLY with the summary text — no intro, no bullet points, no markdown."
)


def _build_summary_message(episodes: list[Episode]) -> str:
    lines = [f"{i + 1}. {ep.content}" for i, ep in enumerate(episodes)]
    return "Summarise these observations into one paragraph:\n" + "\n".join(lines)


def _build_user_message(episodes: list[Episode]) -> str:
    lines = [f"{i + 1}. {ep.content}" for i, ep in enumerate(episodes)]
    return "Extract facts from these observations:\n" + "\n".join(lines)


def _parse_facts_json(text: str) -> list[dict[str, Any]]:
    """Parse LLM JSON response. Returns [] on any parse or validation error."""
    try:
        # Strip markdown fences if the model wraps its output
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = "\n".join(stripped.splitlines()[1:])
            stripped = stripped.rstrip("`").strip()
        data = json.loads(stripped)
        if not isinstance(data, list):
            warnings.warn("Fact extraction returned non-list JSON; ignoring.", stacklevel=2)
            return []
        valid: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict) and {"subject", "predicate", "object"}.issubset(item):
                valid.append(item)
        return valid
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse fact extraction response: %s", exc)
        return []


@runtime_checkable
class LLMAdapter(Protocol):
    """Protocol satisfied by any LLM adapter that can extract facts."""

    model_name: str

    def extract_facts(self, episodes: list[Episode]) -> tuple[list[dict[str, Any]], int]:
        """Extract facts from episodes.

        Returns:
            Tuple of (facts, token_count) where facts is a list of dicts with keys
            subject, predicate, object, confidence, and token_count is the total
            number of tokens consumed by the LLM call (0 if unavailable).
        """
        ...

    def summarise(self, episodes: list[Episode]) -> tuple[str, int]:
        """Summarise a list of episodes into a single paragraph.

        Returns:
            Tuple of (summary_text, token_count).
        """
        ...


class AnthropicAdapter:
    """Uses the Anthropic Messages API for fact extraction.

    Args:
        model: Claude model id to use.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self.model_name = model
        self._client: Any = None

    def _get_client(self) -> Any:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install 'engram[anthropic]'"
            ) from exc
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def extract_facts(self, episodes: list[Episode]) -> tuple[list[dict[str, Any]], int]:
        """Extract facts via Claude. Returns ([], 0) if the LLM response is unparseable."""
        client = self._get_client()
        response = client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(episodes)}],
        )
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return _parse_facts_json(response.content[0].text), tokens

    def summarise(self, episodes: list[Episode]) -> tuple[str, int]:
        """Summarise episodes into one paragraph via Claude."""
        client = self._get_client()
        response = client.messages.create(
            model=self.model_name,
            max_tokens=512,
            system=SUMMARISATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_summary_message(episodes)}],
        )
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return response.content[0].text.strip(), tokens


class OpenAIAdapter:
    """Uses the OpenAI Chat Completions API for fact extraction.

    Also works with Ollama by passing ``base_url="http://localhost:11434/v1"``.

    Args:
        model: Model id to use.
        base_url: Optional custom endpoint (Ollama, Azure, etc.).
    """

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None) -> None:
        self.model_name = model
        self._base_url = base_url
        self._client: Any = None

    def _get_client(self) -> Any:
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "OpenAI SDK not installed. Run: pip install 'engram[openai]'"
            ) from exc
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def extract_facts(self, episodes: list[Episode]) -> tuple[list[dict[str, Any]], int]:
        """Extract facts via OpenAI. Returns ([], 0) if the LLM response is unparseable."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(episodes)},
            ],
        )
        tokens = response.usage.total_tokens if response.usage else 0
        return _parse_facts_json(response.choices[0].message.content or ""), tokens

    def summarise(self, episodes: list[Episode]) -> tuple[str, int]:
        """Summarise episodes into one paragraph via OpenAI."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SUMMARISATION_SYSTEM_PROMPT},
                {"role": "user", "content": _build_summary_message(episodes)},
            ],
        )
        tokens = response.usage.total_tokens if response.usage else 0
        return (response.choices[0].message.content or "").strip(), tokens


class GeminiAdapter:
    """Uses the Google Gemini API for fact extraction and summarisation.

    Reads ``GOOGLE_API_KEY`` from the environment by default.  Pass
    ``api_key`` explicitly to override.

    Args:
        model: Gemini model id (default ``"gemini-2.0-flash"``).
        api_key: Google API key. Falls back to ``GOOGLE_API_KEY`` env var.
    """

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None) -> None:
        self.model_name = model
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        try:
            from google import genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Google Gen AI SDK not installed. Run: pip install 'engram[gemini]'"
            ) from exc
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = genai.Client(**kwargs)
        return self._client

    def extract_facts(self, episodes: list[Episode]) -> tuple[list[dict[str, Any]], int]:
        """Extract facts via Gemini. Returns ([], 0) if the response is unparseable."""
        from google.genai import types  # type: ignore[import-untyped]

        client = self._get_client()
        response = client.models.generate_content(
            model=self.model_name,
            contents=_build_user_message(episodes),
            config=types.GenerateContentConfig(
                system_instruction=EXTRACTION_SYSTEM_PROMPT,
                max_output_tokens=1024,
            ),
        )
        tokens = 0
        if response.usage_metadata:
            tokens = (response.usage_metadata.prompt_token_count or 0) + (
                response.usage_metadata.candidates_token_count or 0
            )
        return _parse_facts_json(response.text or ""), tokens

    def summarise(self, episodes: list[Episode]) -> tuple[str, int]:
        """Summarise episodes into one paragraph via Gemini."""
        from google.genai import types

        client = self._get_client()
        response = client.models.generate_content(
            model=self.model_name,
            contents=_build_summary_message(episodes),
            config=types.GenerateContentConfig(
                system_instruction=SUMMARISATION_SYSTEM_PROMPT,
                max_output_tokens=512,
            ),
        )
        tokens = 0
        if response.usage_metadata:
            tokens = (response.usage_metadata.prompt_token_count or 0) + (
                response.usage_metadata.candidates_token_count or 0
            )
        return (response.text or "").strip(), tokens


class DeepSeekAdapter(OpenAIAdapter):
    """OpenAI-compatible adapter pre-configured for DeepSeek.

    Reads ``DEEPSEEK_API_KEY`` (or ``OPENAI_API_KEY``) from the environment.

    Args:
        model: DeepSeek model id (default ``"deepseek-chat"``).
        api_key: Explicit API key; falls back to env var.
    """

    _BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, model: str = "deepseek-chat", api_key: str | None = None) -> None:
        super().__init__(model=model, base_url=self._BASE_URL)
        self._api_key = api_key

    def _get_client(self) -> Any:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAI SDK not installed. Run: pip install 'engram[openai]'"
            ) from exc
        if self._client is None:
            kwargs: dict[str, Any] = {"base_url": self._BASE_URL}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = openai.OpenAI(**kwargs)
        return self._client


class QwenAdapter(OpenAIAdapter):
    """OpenAI-compatible adapter pre-configured for Alibaba Qwen (DashScope).

    Reads ``DASHSCOPE_API_KEY`` (or ``OPENAI_API_KEY``) from the environment.

    Args:
        model: Qwen model id (default ``"qwen-max"``).
        api_key: Explicit API key; falls back to env var.
    """

    _BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, model: str = "qwen-max", api_key: str | None = None) -> None:
        super().__init__(model=model, base_url=self._BASE_URL)
        self._api_key = api_key

    def _get_client(self) -> Any:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAI SDK not installed. Run: pip install 'engram[openai]'"
            ) from exc
        if self._client is None:
            kwargs: dict[str, Any] = {"base_url": self._BASE_URL}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = openai.OpenAI(**kwargs)
        return self._client


class KimiAdapter(OpenAIAdapter):
    """OpenAI-compatible adapter pre-configured for Moonshot Kimi.

    Reads ``MOONSHOT_API_KEY`` (or ``OPENAI_API_KEY``) from the environment.

    Args:
        model: Kimi model id (default ``"moonshot-v1-8k"``).
        api_key: Explicit API key; falls back to env var.
    """

    _BASE_URL = "https://api.moonshot.cn/v1"

    def __init__(self, model: str = "moonshot-v1-8k", api_key: str | None = None) -> None:
        super().__init__(model=model, base_url=self._BASE_URL)
        self._api_key = api_key

    def _get_client(self) -> Any:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAI SDK not installed. Run: pip install 'engram[openai]'"
            ) from exc
        if self._client is None:
            kwargs: dict[str, Any] = {"base_url": self._BASE_URL}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = openai.OpenAI(**kwargs)
        return self._client


class StubLLMAdapter:
    """Returns pre-configured facts and summaries verbatim. For tests only."""

    model_name: str = "stub"

    def __init__(
        self,
        facts: list[dict[str, Any]] | None = None,
        tokens: int = 0,
        summary: str = "Stub summary.",
    ) -> None:
        self._facts = facts or []
        self._tokens = tokens
        self._summary = summary

    def extract_facts(self, episodes: list[Episode]) -> tuple[list[dict[str, Any]], int]:
        return list(self._facts), self._tokens

    def summarise(self, episodes: list[Episode]) -> tuple[str, int]:
        return self._summary, self._tokens
