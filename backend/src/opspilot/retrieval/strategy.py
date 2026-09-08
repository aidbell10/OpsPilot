"""Single entry point that picks a retrieval strategy and runs it.

Both ``POST /search`` / ``POST /incidents/analyze`` and the evaluation runner
call :func:`retrieve_chunks`, so the three strategies are dispatched in exactly
one place and the API can never drift from what the harness measures.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from opspilot.models.enums import RetrievalStrategy
from opspilot.retrieval.filters import ChunkFilters
from opspilot.retrieval.fusion import DEFAULT_RRF_K
from opspilot.retrieval.hybrid import DEFAULT_CANDIDATE_K, search_hybrid
from opspilot.retrieval.lexical import search_lexical
from opspilot.retrieval.semantic import ChunkMatch, search_chunks


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
) -> list[ChunkMatch]:
    """Retrieve chunks with ``strategy``.

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
