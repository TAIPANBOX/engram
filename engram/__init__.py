"""Engram — cognitive memory layer for AI agents.

See DESIGN.md for architecture and intended API.
"""

from engram.core import Engram
from engram.importance import DecayConfig
from engram.models import Episode, Fact, SearchResult

__version__ = "0.0.1"

__all__ = ["DecayConfig", "Engram", "Episode", "Fact", "SearchResult", "__version__"]
