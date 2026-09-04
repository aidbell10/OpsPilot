from __future__ import annotations

import pytest
from corpus_fixtures import tiny_corpus
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from opspilot.ingestion.pipeline import ingest_corpus
from opspilot.models import Deployment, Document, DocumentChunk, HistoricalIncident, Service
from opspilot.providers.fake import FakeEmbeddingProvider

pytestmark = pytest.mark.integration


def _counts(session: Session) -> dict[str, int]:
    return {
        "services": session.execute(select(func.count()).select_from(Service)).scalar_one(),
        "documents": session.execute(select(func.count()).select_from(Document)).scalar_one(),
        "chunks": session.execute(select(func.count()).select_from(DocumentChunk)).scalar_one(),
        "deployments": session.execute(select(func.count()).select_from(Deployment)).scalar_one(),
        "historical_incidents": session.execute(
            select(func.count()).select_from(HistoricalIncident)
        ).scalar_one(),
    }


def test_ingest_populates_all_tables(db_session: Session, clean_db: None) -> None:
    provider = FakeEmbeddingProvider(dim=384)
    summary = ingest_corpus(
        db_session, tiny_corpus(), embedding_provider=provider, chunk_size=64, chunk_overlap=8
    )
    db_session.commit()

    assert summary.services == 1
    assert summary.documents == 1
    assert summary.chunks >= 1
    assert summary.deployments == 1
    assert summary.historical_incidents == 1
    assert _counts(db_session) == {
        "services": 1,
        "documents": 1,
        "chunks": summary.chunks,
        "deployments": 1,
        "historical_incidents": 1,
    }

    chunk = db_session.execute(select(DocumentChunk)).scalars().first()
    assert chunk is not None
    assert chunk.embedding is not None
    assert len(list(chunk.embedding)) == 384
    assert chunk.service_name == "checkout"

    hist = db_session.execute(select(HistoricalIncident)).scalars().first()
    assert hist is not None
    assert hist.embedding is not None
    assert hist.linked_document_id is not None
    assert hist.meta["incident_id"] == "inc-checkout-500s-01"

    doc = db_session.execute(select(Document)).scalars().first()
    assert doc is not None
    assert doc.service_id is not None


def test_ingest_is_idempotent(db_session: Session, clean_db: None) -> None:
    provider = FakeEmbeddingProvider(dim=384)
    corpus = tiny_corpus()

    ingest_corpus(db_session, corpus, embedding_provider=provider, chunk_size=64, chunk_overlap=8)
    db_session.commit()
    first = _counts(db_session)

    ingest_corpus(db_session, corpus, embedding_provider=provider, chunk_size=64, chunk_overlap=8)
    db_session.commit()
    second = _counts(db_session)

    assert first == second
