"""Cross-encoder reranking of a retrieved candidate list.

Retrieval (vector / lexical / hybrid) is recall-oriented and cheap: it pulls a
generous candidate set with a bi-encoder or an inverted index. A cross-encoder
then re-scores each ``(query, chunk)`` pair jointly — far more accurate at
ordering, far too slow to run over the whole corpus — and the top
``top_k`` survive into the prompt.

This module is the glue only: the model lives behind
:class:`opspilot.providers.base.RerankProvider` (``fake`` or a local
sentence-transformers ``CrossEncoder``). The reranked :class:`ChunkMatch`
objects carry the reranker's ``[0, 1]`` relevance score in ``score``, replacing
the upstream retrieval score.
"""

from __future__ import annotations

from dataclasses import replace

from opspilot.providers.base import RerankProvider
from opspilot.retrieval.semantic import ChunkMatch


def rerank_chunks(
    provider: RerankProvider,
    query: str,
    chunks: list[ChunkMatch],
    *,
    top_k: int,
) -> list[ChunkMatch]:
    """Re-score ``chunks`` against ``query`` and return the best ``top_k``.

    Order is by descending rerank score; ties keep the upstream retrieval
    order, so a reranker that scores everything equally is a no-op.
    """
    if not chunks:
        return []
    scores = provider.rerank(query, [c.content for c in chunks]).scores
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [replace(chunks[i], score=scores[i]) for i in ranked[:top_k]]
