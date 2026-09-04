"""Idempotent upsert of a :class:`~opspilot.ingestion.loader.Corpus` into Postgres.

Idempotency is enforced at the application layer (select-by-natural-key, then
update-or-insert) rather than with database uniqueness constraints, since
ingestion is a single-writer batch job, not concurrent user traffic:

* ``services``             — natural key ``name`` (already ``unique=True`` in the schema)
* ``documents``             — natural key ``source_path``
* ``deployments``           — natural key ``(service_name, version)`` (already unique in the schema)
* ``historical_incidents``  — natural key ``meta["incident_id"]`` (the generator's stable id;
  the ORM model has no dedicated column for it, so it round-trips through the JSONB ``meta``)

A document's chunks are always fully replaced (delete then reinsert) rather than
diffed, since re-chunking is cheap and this guarantees chunk content/embeddings
never drift from the source document.

Historical incidents are embedded on their ``symptoms`` text: that is the same
kind of text an engineer types into ``POST /incidents/analyze`` (a description
of what went wrong), so symptom-to-symptom cosine similarity is the right
analog for "find a similar past incident" at query time — ``root_cause`` and
``resolution`` are only known after investigation and aren't available to
compare against a fresh query.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from opspilot.ingestion.chunking import Chunk, chunk_text
from opspilot.ingestion.loader import (
    Corpus,
    DeploymentRecord,
    DocumentRecord,
    HistoricalIncidentRecord,
    ServiceRecord,
)
from opspilot.models import Deployment, Document, DocumentChunk, HistoricalIncident, Service
from opspilot.models.enums import DeploymentStatus, DocumentType, Environment, Severity
from opspilot.providers.base import EmbeddingProvider


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    services: int
    documents: int
    chunks: int
    deployments: int
    historical_incidents: int


def _upsert_services(session: Session, records: list[ServiceRecord]) -> dict[str, uuid.UUID]:
    service_ids: dict[str, uuid.UUID] = {}
    for rec in records:
        existing = session.execute(
            select(Service).where(Service.name == rec.name)
        ).scalar_one_or_none()
        if existing is None:
            existing = Service(id=uuid.uuid4(), name=rec.name)
            session.add(existing)
        existing.description = rec.description
        existing.repo_url = rec.repo_url
        existing.tier = rec.tier
        existing.owning_team = rec.owning_team
        session.flush()
        service_ids[rec.name] = existing.id
    return service_ids


def _upsert_documents(
    session: Session,
    records: list[DocumentRecord],
    service_ids: dict[str, uuid.UUID],
    embedding_provider: EmbeddingProvider,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[int, int, dict[str, uuid.UUID]]:
    docs: list[Document] = []
    chunk_lists: list[list[Chunk]] = []
    all_texts: list[str] = []

    for rec in records:
        existing = session.execute(
            select(Document).where(Document.source_path == rec.source_path)
        ).scalar_one_or_none()
        if existing is None:
            existing = Document(id=uuid.uuid4(), source_path=rec.source_path)
            session.add(existing)

        existing.document_type = DocumentType(rec.document_type)
        existing.title = rec.title
        existing.content = rec.content
        existing.service_id = service_ids.get(rec.service_name) if rec.service_name else None
        existing.service_name = rec.service_name
        existing.version = rec.version
        existing.environment = Environment(rec.environment) if rec.environment else None
        existing.doc_timestamp = rec.doc_timestamp
        existing.meta = rec.meta
        session.flush()

        session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == existing.id))

        chunks = chunk_text(rec.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs.append(existing)
        chunk_lists.append(chunks)
        all_texts.extend(c.content for c in chunks)

    vectors = embedding_provider.embed(all_texts).vectors if all_texts else []

    doc_ids_by_path: dict[str, uuid.UUID] = {}
    chunk_count = 0
    cursor = 0
    for rec, doc, chunks in zip(records, docs, chunk_lists, strict=True):
        doc_ids_by_path[rec.source_path] = doc.id
        for chunk in chunks:
            session.add(
                DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    chunk_index=chunk.index,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    document_type=doc.document_type,
                    service_name=doc.service_name,
                    version=doc.version,
                    environment=doc.environment,
                    chunk_timestamp=doc.doc_timestamp,
                    meta={},
                    embedding=vectors[cursor],
                )
            )
            cursor += 1
            chunk_count += 1

    session.flush()
    return len(records), chunk_count, doc_ids_by_path


def _upsert_deployments(
    session: Session, records: list[DeploymentRecord], service_ids: dict[str, uuid.UUID]
) -> int:
    for rec in records:
        existing = session.execute(
            select(Deployment).where(
                Deployment.service_name == rec.service_name, Deployment.version == rec.version
            )
        ).scalar_one_or_none()
        if existing is None:
            existing = Deployment(
                id=uuid.uuid4(), service_name=rec.service_name, version=rec.version
            )
            session.add(existing)

        existing.service_id = service_ids.get(rec.service_name)
        existing.environment = Environment(rec.environment)
        existing.deployed_at = rec.deployed_at
        existing.status = DeploymentStatus(rec.status)
        existing.change_summary = rec.change_summary
        existing.changed_components = rec.changed_components
    session.flush()
    return len(records)


def _upsert_historical_incidents(
    session: Session,
    records: list[HistoricalIncidentRecord],
    doc_ids_by_path: dict[str, uuid.UUID],
    embedding_provider: EmbeddingProvider,
) -> int:
    rows: list[HistoricalIncident] = []
    for rec in records:
        existing = session.execute(
            select(HistoricalIncident).where(
                HistoricalIncident.meta["incident_id"].astext == rec.incident_id
            )
        ).scalar_one_or_none()
        if existing is None:
            existing = HistoricalIncident(id=uuid.uuid4())
            session.add(existing)

        existing.title = rec.title
        existing.service_name = rec.service_name
        existing.occurred_at = rec.occurred_at
        existing.severity = Severity(rec.severity)
        existing.symptoms = rec.symptoms
        existing.root_cause = rec.root_cause
        existing.resolution = rec.resolution
        existing.related_deployment_version = rec.related_deployment_version
        existing.linked_document_id = (
            doc_ids_by_path.get(rec.linked_document_source_path)
            if rec.linked_document_source_path
            else None
        )
        existing.meta = {**rec.meta, "incident_id": rec.incident_id}
        rows.append(existing)

    session.flush()

    if records:
        vectors = embedding_provider.embed([rec.symptoms for rec in records]).vectors
        for row, vector in zip(rows, vectors, strict=True):
            row.embedding = vector
        session.flush()

    return len(records)


def ingest_corpus(
    session: Session,
    corpus: Corpus,
    *,
    embedding_provider: EmbeddingProvider,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestionSummary:
    """Upsert an entire corpus. Callers are responsible for committing."""
    service_ids = _upsert_services(session, corpus.services)
    doc_count, chunk_count, doc_ids_by_path = _upsert_documents(
        session,
        corpus.documents,
        service_ids,
        embedding_provider,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    deployment_count = _upsert_deployments(session, corpus.deployments, service_ids)
    historical_count = _upsert_historical_incidents(
        session, corpus.historical_incidents, doc_ids_by_path, embedding_provider
    )
    return IngestionSummary(
        services=len(service_ids),
        documents=doc_count,
        chunks=chunk_count,
        deployments=deployment_count,
        historical_incidents=historical_count,
    )
