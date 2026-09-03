"""Token and variable-cost accounting.

Prices are USD per 1M tokens and are configuration, not measurements — they are
used to estimate ``cost/query`` in evaluation reports. Local embeddings cost
nothing at inference time (compute only), so their price is 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per 1,000,000 tokens. Update as provider pricing changes; the evaluation
# run records which pricebook version produced its numbers.
PRICEBOOK_VERSION = "2026-01"

_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    # model: (input_per_mtok, output_per_mtok)
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "fake-llm-v1": (0.0, 0.0),
}

_DEFAULT_EMBED_PRICES: dict[str, float] = {
    "BAAI/bge-small-en-v1.5": 0.0,  # local
    "fake-embed-v1": 0.0,
}


@dataclass(frozen=True, slots=True)
class PriceBook:
    version: str = PRICEBOOK_VERSION
    llm: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(_DEFAULT_PRICES))
    embedding: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_EMBED_PRICES))

    def llm_cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float:
        in_price, out_price = self.llm.get(model, (0.0, 0.0))
        return (input_tokens * in_price + output_tokens * out_price) / 1_000_000

    def embedding_cost_usd(self, model: str, tokens: int) -> float:
        return tokens * self.embedding.get(model, 0.0) / 1_000_000


def estimate_cost_usd(
    *,
    llm_model: str,
    llm_input_tokens: int,
    llm_output_tokens: int,
    embedding_model: str,
    embedding_tokens: int,
    pricebook: PriceBook | None = None,
) -> float:
    pb = pricebook or PriceBook()
    return pb.llm_cost_usd(llm_model, llm_input_tokens, llm_output_tokens) + pb.embedding_cost_usd(
        embedding_model, embedding_tokens
    )


@dataclass(slots=True)
class CostAccumulator:
    """Mutable tally for a single investigation / evaluation case."""

    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    embedding_tokens: int = 0
    llm_model: str = ""
    embedding_model: str = ""

    def add_llm(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.llm_model = model
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens

    def add_embedding(self, model: str, tokens: int) -> None:
        self.embedding_model = model
        self.embedding_tokens += tokens

    def total_usd(self, pricebook: PriceBook | None = None) -> float:
        return estimate_cost_usd(
            llm_model=self.llm_model,
            llm_input_tokens=self.llm_input_tokens,
            llm_output_tokens=self.llm_output_tokens,
            embedding_model=self.embedding_model,
            embedding_tokens=self.embedding_tokens,
            pricebook=pricebook,
        )

    def as_dict(self, pricebook: PriceBook | None = None) -> dict[str, float | str]:
        return {
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "embedding_tokens": self.embedding_tokens,
            "estimated_usd": round(self.total_usd(pricebook), 8),
            "pricebook_version": (pricebook or PriceBook()).version,
        }
