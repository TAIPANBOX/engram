"""Importance scoring formula and decay configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DecayConfig:
    """Hyper-parameters controlling the importance scoring formula.

    importance(m, t) =
        salience * exp(-lambda_ * elapsed_days)
      + alpha * log(1 + access_count)
      + beta * emotional_valence
    """

    lambda_: float = 0.1   # decay rate (per day); half-life ≈ 7 days
    alpha: float = 0.2     # reinforcement weight from access frequency
    beta: float = 0.1      # emotional weight
    threshold: float = 0.1 # minimum importance before pruning (used in reflection, v0.3)


def compute_importance(
    salience: float,
    emotional_valence: float,
    last_access: datetime,
    access_count: int,
    now: datetime,
    cfg: DecayConfig,
) -> float:
    """Compute the importance score for a memory at time *now*.

    Args:
        salience: Subjective weight at encoding time (0-1).
        emotional_valence: Affective tag (-1 to +1).
        last_access: When the memory was last accessed (or created if never accessed).
        access_count: Number of times the memory has been retrieved.
        now: Reference point in time for computing elapsed days.
        cfg: Decay hyper-parameters.

    Returns:
        Importance score (float, lower-bounded only by the formula; can be negative
        if emotional_valence is strongly negative and salience/access are low).
    """
    elapsed_days = (now - last_access).total_seconds() / 86400.0
    decay_term = salience * math.exp(-cfg.lambda_ * elapsed_days)
    access_term = cfg.alpha * math.log1p(access_count)
    emotion_term = cfg.beta * emotional_valence
    return decay_term + access_term + emotion_term
