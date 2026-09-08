"""Single entry point that picks a retrieval strategy, runs it, and optionally reranks.

Both ``POST /search`` / ``POST /incidents/analyze`` and the evaluation runner
retrieve through here, so strategy dispatch and the rerank stage are defined in
exactly one place and the API can never drift from what the harness measures.
(The runner calls the pieces one level lower — see
:mod:`opspilot.evaluation.runner` — only to capture per-stage latency.)
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from opspilot.models.enums import RetrievalStrategy
from opspilot.providers.base import RerankProvider
from opspilot.retrieval.filters import ChunkFilters
from opspilot.retrieval.fusion import DEFAULT_RRF_K
from opspilot.retrieval.hybrid import DEFAULT_CANDIDATE_K, search_hybrid
from opspilot.retrieval.lexical import search_lexical
from opspilot.retrieval.rerank import rerank_chunks
from opspilot.retrieval.semantic import ChunkMatch, search_chunks


def retrieve_candidates(
    session: Session,
    *,
    strategy: RetrievalStrategy,
    query_embedding: list[float],
    query_text: str,
    top_k: int,
    filters: ChunkFilters | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[ChunkMatch]:
    """Run ``strategy`` and return its top ``top_k`` chunks (no reranking).

    ``query_embedding`` is ignored by the pure-lexical strategy and
    ``query_text`` by the pure-vector strategy; callers pass both so the choice
    stays a one-line switch.
    """
    if strategy is RetrievalStrategy.VECTOR:
        return search_chunks(session, query_embedding, top_k=top_k, filters=filters)
    if strategy is RetrievalStrategy.LEXICAL:
        return search_lexical(session, query_text, top_k=top_k, filters=filters)
    return search_hybrid(
        session,
        query_embedding=query_embedding,
        query_text=query_text,
        top_k=top_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        filters=filters,
    )


def retrieve_chunks(
    session: Session,
    *,
    strategy: RetrievalStrategy,
    query_embedding: list[float],
    query_text: str,
    top_k: int,
    filters: ChunkFilters | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rrf_k: int = DEFAULT_RRF_K,
    reranker: RerankProvider | None = None,
    rerank_candidate_k: int = 20,
) -> list[ChunkMatch]:
    """Retrieve, and rerank when ``reranker`` is given.

    With a reranker, ``max(top_k, rerank_candidate_k)`` candidates are fetched
    and the cross-encoder trims them back to ``top_k``.
    """
    fetch_k = max(top_k, rerank_candidate_k) if reranker is not None else top_k
    candidates = retrieve_candidates(
        session,
        strategy=strategy,
        query_embedding=query_embedding,
        query_text=query_text,
        top_k=fetch_k,
        filters=filters,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )
    if reranker is None:
        return candidates
    return rerank_chunks(reranker, query_text, candidates, top_k=top_k)
