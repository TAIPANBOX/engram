"""Engram — cognitive memory layer for AI agents.

See DESIGN.md for architecture and intended API.
"""

from engram.core import Engram, ReflectionThread
from engram.importance import DecayConfig
from engram.llm import AnthropicAdapter, OpenAIAdapter, StubLLMAdapter
from engram.models import Episode, Fact, ReflectionRun, SearchResult

__version__ = "1.0.0"

__all__ = [
    "AnthropicAdapter",
    "DecayConfig",
    "Engram",
    "Episode",
    "Fact",
    "OpenAIAdapter",
    "ReflectionRun",
    "ReflectionThread",
    "SearchResult",
    "StubLLMAdapter",
    "__version__",
]
