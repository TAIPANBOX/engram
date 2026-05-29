"""Engram — cognitive memory layer for AI agents.

See DESIGN.md for architecture and intended API.
"""

from engram.async_engram import AsyncEngram
from engram.core import Engram, ReflectionThread
from engram.importance import DecayConfig
from engram.llm import (
    AnthropicAdapter,
    DeepSeekAdapter,
    GeminiAdapter,
    KimiAdapter,
    OpenAIAdapter,
    QwenAdapter,
    StubLLMAdapter,
)
from engram.models import (
    CompressionRun,
    Episode,
    Fact,
    ForgetResult,
    ObserveInput,
    ReflectionRun,
    SearchResult,
)
from engram.working_memory import WorkingMemory, WorkingMemoryItem

__version__ = "2.1.2"

__all__ = [
    "AnthropicAdapter",
    "AsyncEngram",
    "CompressionRun",
    "DecayConfig",
    "DeepSeekAdapter",
    "Engram",
    "Episode",
    "Fact",
    "ForgetResult",
    "GeminiAdapter",
    "KimiAdapter",
    "ObserveInput",
    "OpenAIAdapter",
    "QwenAdapter",
    "ReflectionRun",
    "ReflectionThread",
    "SearchResult",
    "StubLLMAdapter",
    "WorkingMemory",
    "WorkingMemoryItem",
    "__version__",
]
