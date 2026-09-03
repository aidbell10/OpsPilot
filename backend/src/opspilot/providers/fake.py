"""Deterministic offline providers.

Used by default in tests and CI. No network, no model download, no cost. The
embeddings are not semantically meaningful, but they ARE:

* deterministic  — identical input always yields an identical vector
* normalised     — unit L2 norm, like a real sentence-transformer
* lightly lexical — vectors derived from token hashes, so texts that share
  tokens land closer together than texts that don't (enough for wiring tests)
"""

from __future__ import annotations

import hashlib
import math
import re

from opspilot.providers.base import EmbeddingResult, LLMResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hash_floats(seed: str, count: int) -> list[float]:
    """Expand a seed string into ``count`` deterministic floats in [-1, 1)."""
    out: list[float] = []
    counter = 0
    while len(out) < count:
        digest = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        for i in range(0, len(digest), 4):
            if len(out) >= count:
                break
            n = int.from_bytes(digest[i : i + 4], "big")
            out.append((n / 2**31) - 1.0)
        counter += 1
    return out


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FakeEmbeddingProvider:
    name = "fake"

    def __init__(self, model: str = "fake-embed-v1", dim: int = 384) -> None:
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> EmbeddingResult:
        vectors: list[list[float]] = []
        total_tokens = 0
        for text in texts:
            tokens = _TOKEN_RE.findall(text.lower())
            total_tokens += len(tokens)
            acc = [0.0] * self.dim
            for tok in tokens or ["<empty>"]:
                for i, v in enumerate(_hash_floats(tok, self.dim)):
                    acc[i] += v
            vectors.append(_normalise(acc))
        return EmbeddingResult(vectors=vectors, model=self.model, total_tokens=total_tokens)


class FakeLLMProvider:
    name = "fake"

    def __init__(self, model: str = "fake-llm-v1") -> None:
        self.model = model

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> LLMResult:
        # Deterministic, obviously-synthetic echo. The generation layer treats
        # this as a black box; tests assert on structure, not content.
        digest = hashlib.sha256(f"{system}\n---\n{user}".encode()).hexdigest()[:12]
        text = (
            "[fake-llm] deterministic response "
            f"({digest}). This provider does not reason; configure "
            "OPSPILOT_LLM_PROVIDER=anthropic for real generation."
        )
        return LLMResult(
            text=text,
            model=self.model,
            input_tokens=len(system.split()) + len(user.split()),
            output_tokens=len(text.split()),
            stop_reason="end_turn",
        )
