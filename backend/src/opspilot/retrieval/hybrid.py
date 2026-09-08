"""Hybrid retrieval: the vector and lexical arms combined with RRF.

Each arm retrieves ``candidate_k`` chunks; their two rankings are fused with
Reciprocal Rank Fusion (:mod:`opspilot.retrieval.fusion`) and the top
``top_k`` survivors are returned. Fusion needs only rank position, so the
cosine-similarity scale and the ``ts_rank_cd`` scale never have to be
reconciled.

The returned :class:`~opspilot.retrieval.semantic.ChunkMatch` objects carry the
RRF score in ``score`` (not the originating arm's score), and the metadata /
content is taken from whichever arm first produced that chunk — the fields are
identical across arms since both read the same ``document_chunks`` row.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

from sqlalchemy.orm import Session

from opspilot.retrieval.filters import ChunkFilters
from opspilot.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from opspilot.retrieval.lexical import search_lexical
from opspilot.retrieval.semantic import ChunkMatch, search_chunks

DEFAULT_CANDIDATE_K = 30


def search_hybrid(
    session: Session,
    *,
    query_embedding: list[float],
    query_text: str,
    top_k: int,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rrf_k: int = DEFAULT_RRF_K,
    filters: ChunkFilters | None = None,
) -> list[ChunkMatch]:
    filters = filters or ChunkFilters()
    vector_hits = search_chunks(session, query_embedding, top_k=candidate_k, filters=filters)
    lexical_hits = search_lexical(session, query_text, top_k=candidate_k, filters=filters)

    by_id: dict[uuid.UUID, ChunkMatch] = {}
    for hit in (*vector_hits, *lexical_hits):
        by_id.setdefault(hit.chunk_id, hit)

    fused = reciprocal_rank_fusion(
        [[h.chunk_id for h in vector_hits], [h.chunk_id for h in lexical_hits]],
        k=rrf_k,
    )
    return [replace(by_id[chunk_id], score=rrf_score) for chunk_id, rrf_score in fused[:top_k]]
