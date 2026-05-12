"""Engram — cognitive memory layer for AI agents.

See DESIGN.md for architecture and intended API.
"""

from engram.core import Engram, ReflectionThread
from engram.importance import DecayConfig
from engram.llm import AnthropicAdapter, OpenAIAdapter, StubLLMAdapter
from engram.models import Episode, Fact, ForgetResult, ObserveInput, ReflectionRun, SearchResult

__version__ = "2.0.1"

__all__ = [
    "AnthropicAdapter",
    "DecayConfig",
    "Engram",
    "Episode",
    "Fact",
    "ForgetResult",
    "ObserveInput",
    "OpenAIAdapter",
    "ReflectionRun",
    "ReflectionThread",
    "SearchResult",
    "StubLLMAdapter",
    "__version__",
]
