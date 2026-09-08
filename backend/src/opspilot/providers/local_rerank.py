"""Local sentence-transformers cross-encoder reranker.

Requires the ``ml`` optional dependency group (``uv sync --extra ml``). The
model is loaded lazily on first use and cached for the process lifetime.

The default model, ``cross-encoder/ms-marco-MiniLM-L-6-v2``, is a 6-layer
MiniLM trained on MS MARCO passage ranking - small enough to run on CPU over a
20-to-30 candidate list in tens of milliseconds. It outputs an unbounded
relevance logit per pair; this wrapper applies a logistic squash so callers
get a ``[0, 1]`` relevance estimate (see :class:`opspilot.providers.base.RerankResult`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from opspilot.providers.base import RerankResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


def _sigmoid(x: float) -> float:
    # Guard against overflow for large-magnitude logits.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class LocalCrossEncoderProvider:
    name = "cross_encoder"

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = model
        self._ce: CrossEncoder | None = None

    def _model(self) -> CrossEncoder:
        if self._ce is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - exercised via message
                raise RuntimeError(
                    "LocalCrossEncoderProvider needs the 'ml' extra: run `uv sync --extra ml` "
                    "or set OPSPILOT_RERANKER=none."
                ) from exc
            self._ce = CrossEncoder(self.model)
        return self._ce

    def rerank(self, query: str, documents: list[str]) -> RerankResult:
        if not documents:
            return RerankResult(scores=[], model=self.model)
        raw = self._model().predict(
            [(query, doc) for doc in documents],
            show_progress_bar=False,
        )
        return RerankResult(scores=[_sigmoid(float(x)) for x in raw], model=self.model)
