from __future__ import annotations

import datetime as dt
import uuid

import pytest
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.orm import Session

from opspilot.models import Document, DocumentChunk
from opspilot.models.enums import DocumentType, Environment
from opspilot.providers.fake import FakeEmbeddingProvider

pytestmark = pytest.mark.integration

_EXPECTED_TABLES = {
    "services",
    "documents",
    "document_chunks",
    "deployments",
    "incidents",
    "historical_incidents",
    "evaluation_cases",
    "evaluation_runs",
    "evaluation_results",
    "user_feedback",
    "alembic_version",
}


def test_schema_has_all_tables(migrated_engine: Engine) -> None:
    tables = set(inspect(migrated_engine).get_table_names())
    assert tables >= _EXPECTED_TABLES


def test_pgvector_extension_installed(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as conn:
        row = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).first()
    assert row is not None


def test_hnsw_and_gin_indexes_exist(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as conn:
        idx = {
            r[0]
            for r in conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = 'document_chunks'")
            )
        }
    assert "ix_document_chunks_embedding_hnsw" in idx
    assert "ix_document_chunks_content_tsv" in idx


def test_document_chunk_roundtrip_with_embedding(db_session: Session, clean_db: None) -> None:
    embedder = FakeEmbeddingProvider(dim=384)
    doc = Document(
        id=uuid.uuid4(),
        document_type=DocumentType.RUNBOOK,
        title="Checkout 500s troubleshooting",
        content="If checkout returns HTTP 500 after a deploy, check promotion validation.",
        service_name="checkout-service",
        version="v2.14.0",
        environment=Environment.PRODUCTION,
    )
    chunk = DocumentChunk(
        id=uuid.uuid4(),
        document=doc,
        chunk_index=0,
        content=doc.content,
        token_count=15,
        document_type=DocumentType.RUNBOOK,
        service_name="checkout-service",
        version="v2.14.0",
        environment=Environment.PRODUCTION,
        chunk_timestamp=dt.datetime(2026, 8, 1, tzinfo=dt.UTC),
        embedding=embedder.embed([doc.content]).vectors[0],
    )
    db_session.add(doc)
    db_session.add(chunk)
    db_session.commit()

    fetched = db_session.get(DocumentChunk, chunk.id)
    assert fetched is not None
    assert fetched.document_type is DocumentType.RUNBOOK
    assert fetched.embedding is not None
    assert len(list(fetched.embedding)) == 384

    # generated tsvector is populated by the database
    tsv = db_session.execute(
        text("SELECT content_tsv::text FROM document_chunks WHERE id = :i"), {"i": chunk.id}
    ).scalar_one()
    assert "checkout" in tsv

    # cosine distance operator works against the stored vector
    probe = embedder.embed(["checkout HTTP 500 promotion validation"]).vectors[0]
    distance = db_session.execute(
        select(DocumentChunk.embedding.cosine_distance(probe)).where(DocumentChunk.id == chunk.id)
    ).scalar_one()
    assert 0.0 <= float(distance) <= 2.0
