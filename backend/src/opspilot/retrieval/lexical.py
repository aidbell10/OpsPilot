"""Lexical retrieval over ``document_chunks.content_tsv`` (PostgreSQL FTS).

Full-text search complements vector search: it nails the exact tokens a
paraphrase-trained embedding blurs together — error codes (``PAY-50231``),
function names (``authorize_payment``), version strings (``v5.0.6``). The
``content_tsv`` column is ``GENERATED ALWAYS AS to_tsvector('english', content)``
with a GIN index (both created in migration 0001), so this is an index scan,
not a per-query re-parse of the corpus.

User text is parsed by ``websearch_to_tsquery`` first — it takes raw input
(quoted phrases, ``or``, leading ``-`` negation) and never raises on
punctuation the way ``to_tsquery`` does — but that function ``AND``\\ s every
term, which is far too strict for a natural-language incident description
(demanding one chunk contain *every* word kills recall). So its parsed output
is re-joined with ``|`` (``to_tsquery(replace(… ' & ' → ' | '))``): any term
may match, and ``ts_rank_cd`` sorts by how many / how densely they do. Phrase
(``<->``) and negation (``!``) sub-expressions are left intact.

``ts_rank_cd`` (cover density — rewards matches that sit close together) is
unbounded and not comparable across queries, so it is squashed to ``(0, 1]``
with ``r / (r + 1)`` purely for display; fusion
(:mod:`opspilot.retrieval.fusion`) consumes rank *position*, not this score.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Function, Text, cast, func, select
from sqlalchemy.orm import Session

from opspilot.models.document import Document, DocumentChunk
from opspilot.retrieval.filters import ChunkFilters
from opspilot.retrieval.semantic import ChunkMatch


def _rank_to_score(rank: float) -> float:
    """Map an unbounded ``ts_rank_cd`` value to ``(0, 1]`` for display."""
    return rank / (rank + 1.0) if rank > 0 else 0.0


def _or_tsquery(query_text: str) -> Function[Any]:
    """A ``tsquery`` that OR-joins the top-level terms of ``query_text``."""
    parsed = func.websearch_to_tsquery("english", query_text)
    return func.to_tsquery("english", func.replace(cast(parsed, Text), " & ", " | "))


def search_lexical(
    session: Session,
    query_text: str,
    *,
    top_k: int,
    filters: ChunkFilters | None = None,
) -> list[ChunkMatch]:
    """Top-``top_k`` chunks whose full-text vector matches ``query_text``.

    Chunks with no lexical overlap are excluded entirely (``@@`` predicate), so
    unlike the vector arm this can return fewer than ``top_k`` — or nothing.
    """
    tsquery = _or_tsquery(query_text)
    rank = func.ts_rank_cd(DocumentChunk.content_tsv, tsquery)
    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.content,
            DocumentChunk.document_type,
            DocumentChunk.service_name,
            DocumentChunk.version,
            Document.title,
            Document.source_path,
            rank.label("rank"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc(), DocumentChunk.id)
        .limit(top_k)
    )
    for clause in (filters or ChunkFilters()).clauses():
        stmt = stmt.where(clause)

    rows = session.execute(stmt).all()
    return [
        ChunkMatch(
            chunk_id=row.id,
            document_id=row.document_id,
            content=row.content,
            score=_rank_to_score(float(row.rank)),
            title=row.title,
            source_path=row.source_path,
            document_type=row.document_type,
            service_name=row.service_name,
            version=row.version,
        )
        for row in rows
    ]
