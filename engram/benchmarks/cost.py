"""Reflection cost benchmark: estimates token usage and projected LLM spend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engram.llm import EXTRACTION_SYSTEM_PROMPT, _build_user_message
from engram.models import Episode

# Model pricing table (USD per million tokens, as of mid-2025).
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4.5": (0.25, 1.25),
    "claude-sonnet-4.6": (3.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


@dataclass
class CostResult:
    """Token usage and cost projection for a reflection benchmark run."""

    n_episodes: int
    n_reflect_runs: int
    facts_extracted: int
    est_input_tokens: int
    est_output_tokens: int

    @property
    def tokens_per_1000_episodes(self) -> float:
        total = self.est_input_tokens + self.est_output_tokens
        return total / self.n_episodes * 1000 if self.n_episodes else 0.0

    def cost_usd(self, model: str = "claude-haiku-4.5") -> float:
        """Estimated USD cost for this run at *model* pricing."""
        inp_price, out_price = _MODEL_PRICES.get(model, (1.0, 5.0))
        return self.est_input_tokens / 1e6 * inp_price + self.est_output_tokens / 1e6 * out_price

    def projected_cost_per_1000_episodes(self, model: str = "claude-haiku-4.5") -> float:
        inp_price, out_price = _MODEL_PRICES.get(model, (1.0, 5.0))
        inp = self.est_input_tokens / self.n_episodes * 1000 if self.n_episodes else 0.0
        out = self.est_output_tokens / self.n_episodes * 1000 if self.n_episodes else 0.0
        return inp / 1e6 * inp_price + out / 1e6 * out_price

    def __str__(self) -> str:
        lines = [
            f"episodes={self.n_episodes}  runs={self.n_reflect_runs}  facts={self.facts_extracted}",
            f"  est_input_tokens={self.est_input_tokens:,}  "
            f"est_output_tokens={self.est_output_tokens:,}",
            f"  tokens_per_1000_ep={self.tokens_per_1000_episodes:,.0f}",
        ]
        return "\n".join(lines)


class _CountingAdapter:
    """LLM adapter that counts estimated tokens without calling any API."""

    model_name: str = "cost-counter"

    def __init__(self, facts_per_batch: int = 3) -> None:
        self._facts_per_batch = facts_per_batch
        self.input_tokens = 0
        self.output_tokens = 0

    def extract_facts(self, episodes: list[Episode]) -> list[dict[str, Any]]:
        msg = EXTRACTION_SYSTEM_PROMPT + _build_user_message(episodes)
        # Approximate: 1 token ≈ 4 characters (GPT tokenizer heuristic)
        self.input_tokens += len(msg) // 4
        # Each fact JSON object ≈ 60 tokens
        self.output_tokens += self._facts_per_batch * 60
        return [
            {
                "subject": f"entity_{i}",
                "predicate": "has_property",
                "object": f"value_{i}",
                "confidence": 0.8,
            }
            for i in range(self._facts_per_batch)
        ]


def run_cost_bench(
    n_episodes: int = 200,
    facts_per_batch: int = 3,
    batch_size: int = 20,
) -> CostResult:
    """Estimate reflection token usage by simulating N episodes.

    Uses a non-LLM adapter that counts tokens from prompt construction
    without making any API calls.

    Args:
        n_episodes: Total episodes to process.
        facts_per_batch: Synthetic facts the mock LLM "extracts" per batch.
        batch_size: Episodes per reflection call (controlled by store limit).

    Returns:
        :class:`CostResult` with token estimates and cost projections.
    """
    from engram.core import Engram
    from engram.importance import DecayConfig

    adapter = _CountingAdapter(facts_per_batch=facts_per_batch)
    # High threshold so nothing gets pruned during the bench
    cfg = DecayConfig(threshold=0.0)
    mem = Engram(llm=adapter, decay_config=cfg)

    # Insert all episodes
    for i in range(n_episodes):
        mem.observe(
            f"Benchmark episode {i}: entity_{i % 10} performed action_{i % 5} at location_{i % 3}."
        )

    # Reflect in batches until all episodes are covered
    n_runs = 0
    while True:
        run = mem.reflect()
        n_runs += 1
        # Stop when no new episodes are processed (all already reflected)
        if run.episodes_processed == 0:
            break
        if n_runs >= n_episodes:  # safety cap
            break

    return CostResult(
        n_episodes=n_episodes,
        n_reflect_runs=n_runs,
        facts_extracted=adapter._facts_per_batch * n_runs,
        est_input_tokens=adapter.input_tokens,
        est_output_tokens=adapter.output_tokens,
    )
