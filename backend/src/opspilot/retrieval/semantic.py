"""Vector-only semantic retrieval over ``document_chunks`` and ``historical_incidents``.

Lexical (Phase 5) and reranking (Phase 6/7) arms are added later without
touching this module's shape — callers already get a plain ranked list.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from opspilot.models.document import Document, DocumentChunk
from opspilot.models.enums import DocumentType
from opspilot.models.historical_incident import HistoricalIncident
from opspilot.providers.base import EmbeddingProvider


def _distance_to_score(distance: float) -> float:
    """Cosine distance in ``[0, 2]`` -> a similarity score in ``[0, 1]``."""
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


@dataclass(frozen=True, slots=True)
class ChunkMatch:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    score: float
    title: str
    source_path: str | None
    document_type: DocumentType
    service_name: str | None
    version: str | None


@dataclass(frozen=True, slots=True)
class HistoricalMatch:
    incident_id: uuid.UUID
    title: str
    service_name: str
    occurred_at: str
    severity: str
    root_cause: str
    resolution: str
    score: float


def search_chunks(
    session: Session,
    query_embedding: list[float],
    *,
    top_k: int,
    service_name: str | None = None,
    document_type: DocumentType | None = None,
) -> list[ChunkMatch]:
    """Top-``top_k`` chunks by cosine distance, with optional metadata filters."""
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
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
            distance.label("distance"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(distance)
        .limit(top_k)
    )
    if service_name is not None:
        stmt = stmt.where(DocumentChunk.service_name == service_name)
    if document_type is not None:
        stmt = stmt.where(DocumentChunk.document_type == document_type)

    rows = session.execute(stmt).all()
    return [
        ChunkMatch(
            chunk_id=row.id,
            document_id=row.document_id,
            content=row.content,
            score=_distance_to_score(float(row.distance)),
            title=row.title,
            source_path=row.source_path,
            document_type=row.document_type,
            service_name=row.service_name,
            version=row.version,
        )
        for row in rows
    ]


def search_historical_incidents(
    session: Session,
    query_embedding: list[float],
    *,
    top_k: int,
    service_name: str | None = None,
) -> list[HistoricalMatch]:
    """Top-``top_k`` historical incidents by symptom-embedding cosine distance."""
    distance = HistoricalIncident.embedding.cosine_distance(query_embedding)
    stmt = (
        select(HistoricalIncident, distance.label("distance"))
        .where(HistoricalIncident.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    if service_name is not None:
        stmt = stmt.where(HistoricalIncident.service_name == service_name)

    rows = session.execute(stmt).all()
    return [
        HistoricalMatch(
            incident_id=incident.id,
            title=incident.title,
            service_name=incident.service_name,
            occurred_at=incident.occurred_at.isoformat(),
            severity=str(incident.severity.value),
            root_cause=incident.root_cause,
            resolution=incident.resolution,
            score=_distance_to_score(float(dist)),
        )
        for incident, dist in rows
    ]


def embed_and_search_chunks(
    session: Session,
    embedding_provider: EmbeddingProvider,
    query: str,
    *,
    top_k: int,
    service_name: str | None = None,
    document_type: DocumentType | None = None,
) -> list[ChunkMatch]:
    vector = embedding_provider.embed([query]).vectors[0]
    return search_chunks(
        session, vector, top_k=top_k, service_name=service_name, document_type=document_type
    )


def embed_and_search_historical_incidents(
    session: Session,
    embedding_provider: EmbeddingProvider,
    query: str,
    *,
    top_k: int,
    service_name: str | None = None,
) -> list[HistoricalMatch]:
    vector = embedding_provider.embed([query]).vectors[0]
    return search_historical_incidents(session, vector, top_k=top_k, service_name=service_name)
