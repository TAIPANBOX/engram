"""Tests for the importance scoring formula."""

import math
from datetime import UTC, datetime, timedelta

from engram.importance import DecayConfig, compute_importance

_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
_CFG = DecayConfig(lambda_=0.1, alpha=0.2, beta=0.1, threshold=0.1)


def test_zero_elapsed_zero_accesses() -> None:
    score = compute_importance(
        salience=0.8,
        emotional_valence=0.0,
        last_access=_NOW,
        access_count=0,
        now=_NOW,
        cfg=_CFG,
    )
    # decay term = 0.8 * exp(0) = 0.8; access = 0; emotion = 0
    assert math.isclose(score, 0.8, rel_tol=1e-6)


def test_emotional_weight_contributes() -> None:
    score_pos = compute_importance(0.5, 1.0, _NOW, 0, _NOW, _CFG)
    score_neg = compute_importance(0.5, -1.0, _NOW, 0, _NOW, _CFG)
    score_zero = compute_importance(0.5, 0.0, _NOW, 0, _NOW, _CFG)
    assert score_pos > score_zero > score_neg
    assert math.isclose(score_pos - score_zero, _CFG.beta, rel_tol=1e-6)


def test_decay_reduces_score_over_time() -> None:
    last = _NOW - timedelta(days=30)
    score_recent = compute_importance(0.8, 0.0, _NOW - timedelta(days=1), 0, _NOW, _CFG)
    score_old = compute_importance(0.8, 0.0, last, 0, _NOW, _CFG)
    assert score_recent > score_old


def test_access_reinforcement_increases_score() -> None:
    base = compute_importance(0.5, 0.0, _NOW, 0, _NOW, _CFG)
    reinforced = compute_importance(0.5, 0.0, _NOW, 10, _NOW, _CFG)
    assert reinforced > base


def test_access_reinforcement_logarithmic() -> None:
    # Marginal gain per +1 access shrinks as count grows (diminishing returns)
    gain_first = (
        compute_importance(0.5, 0.0, _NOW, 1, _NOW, _CFG)
        - compute_importance(0.5, 0.0, _NOW, 0, _NOW, _CFG)
    )
    gain_tenth = (
        compute_importance(0.5, 0.0, _NOW, 10, _NOW, _CFG)
        - compute_importance(0.5, 0.0, _NOW, 9, _NOW, _CFG)
    )
    assert gain_first > gain_tenth


def test_full_formula_composition() -> None:
    last = _NOW - timedelta(days=7)
    score = compute_importance(
        salience=0.6,
        emotional_valence=0.5,
        last_access=last,
        access_count=3,
        now=_NOW,
        cfg=_CFG,
    )
    expected = (
        0.6 * math.exp(-0.1 * 7.0)
        + 0.2 * math.log1p(3)
        + 0.1 * 0.5
    )
    assert math.isclose(score, expected, rel_tol=1e-6)
