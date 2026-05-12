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

    def extract_facts(self, episodes: list[Episode]) -> list[dict[str, Any]]:
        """Return a list of raw fact dicts with keys subject, predicate, object, confidence."""
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

    def extract_facts(self, episodes: list[Episode]) -> list[dict[str, Any]]:
        """Extract facts via Claude. Returns [] if the LLM response is unparseable."""
        client = self._get_client()
        response = client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(episodes)}],
        )
        return _parse_facts_json(response.content[0].text)


class OpenAIAdapter:
    """Uses the OpenAI Chat Completions API for fact extraction.

    Also works with Ollama by passing ``base_url="http://localhost:11434/v1"``.

    Args:
        model: Model id to use.
        base_url: Optional custom endpoint (Ollama, Azure, etc.).
    """

    def __init__(
        self, model: str = "gpt-4o-mini", base_url: str | None = None
    ) -> None:
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

    def extract_facts(self, episodes: list[Episode]) -> list[dict[str, Any]]:
        """Extract facts via OpenAI. Returns [] if the LLM response is unparseable."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(episodes)},
            ],
        )
        return _parse_facts_json(response.choices[0].message.content or "")


class StubLLMAdapter:
    """Returns pre-configured facts verbatim. For tests only."""

    model_name: str = "stub"

    def __init__(self, facts: list[dict[str, Any]] | None = None) -> None:
        self._facts = facts or []

    def extract_facts(self, episodes: list[Episode]) -> list[dict[str, Any]]:
        return list(self._facts)
