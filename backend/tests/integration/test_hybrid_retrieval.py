"""Phase 5 — lexical FTS, metadata filters, and RRF hybrid fusion against a live DB."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.orm import Session

from opspilot.ingestion.loader import Corpus, DocumentRecord, ServiceRecord
from opspilot.ingestion.pipeline import ingest_corpus
from opspilot.models.enums import DocumentType
from opspilot.providers.fake import FakeEmbeddingProvider
from opspilot.retrieval.filters import ChunkFilters
from opspilot.retrieval.hybrid import search_hybrid
from opspilot.retrieval.lexical import search_lexical
from opspilot.retrieval.semantic import search_chunks

pytestmark = [pytest.mark.integration, pytest.mark.retrieval]

_QUERY = "payments error PAY-50231 from authorize_payment gateway timeout"


def _corpus() -> Corpus:
    body = (
        "The authorize_payment function calls the downstream card gateway. "
        "When the gateway is slow it raises a timeout and the charge is retried. "
    )
    return Corpus(
        services=[
            ServiceRecord(name="payments", description="charges", tier=1, owning_team="fin"),
            ServiceRecord(name="inventory", description="stock", tier=2, owning_team="ops"),
        ],
        documents=[
            DocumentRecord(
                doc_id="rb-pay-timeouts",
                document_type="runbook",
                title="Payments gateway timeouts",
                content=body * 6,
                service_name="payments",
                version="v5.0.6",
                environment="production",
                source_path="runbooks/payments/timeouts.md",
                doc_timestamp=dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
            ),
            DocumentRecord(
                doc_id="ke-pay-50231",
                document_type="known_error",
                title="PAY-50231 authorization declined",
                content=(
                    "Error code PAY-50231 is emitted by authorize_payment when the gateway "
                    "returns a hard decline. PAY-50231 is not retryable. "
                )
                * 6,
                service_name="payments",
                version="v5.0.6",
                environment="production",
                source_path="known-errors/payments/PAY-50231.md",
                doc_timestamp=dt.datetime(2025, 3, 2, tzinfo=dt.UTC),
            ),
            DocumentRecord(
                doc_id="rb-inv-restock",
                document_type="runbook",
                title="Inventory restock job",
                content=(
                    "The nightly restock job reconciles warehouse counts against supplier "
                    "manifests and reorders items below their threshold. "
                )
                * 6,
                service_name="inventory",
                version="v3.1.0",
                environment="production",
                source_path="runbooks/inventory/restock.md",
                doc_timestamp=dt.datetime(2025, 3, 3, tzinfo=dt.UTC),
            ),
        ],
        deployments=[],
        historical_incidents=[],
    )


@pytest.fixture
def seeded(db_session: Session, clean_db: None) -> Session:
    ingest_corpus(
        db_session,
        _corpus(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        chunk_size=64,
        chunk_overlap=8,
    )
    db_session.commit()
    return db_session


def test_lexical_finds_exact_error_code_and_excludes_unrelated(seeded: Session) -> None:
    hits = search_lexical(seeded, _QUERY, top_k=10)
    paths = {m.source_path for m in hits}
    assert "known-errors/payments/PAY-50231.md" in paths
    # the inventory runbook shares no query tokens -> the @@ predicate drops it
    assert "runbooks/inventory/restock.md" not in paths
    assert all(0.0 < m.score <= 1.0 for m in hits)


def test_lexical_returns_nothing_when_no_chunk_matches(seeded: Session) -> None:
    assert search_lexical(seeded, "xyzzy plugh frobnicate", top_k=10) == []


def test_metadata_filters_apply_to_lexical(seeded: Session) -> None:
    known_errors = search_lexical(
        seeded, _QUERY, top_k=10, filters=ChunkFilters(document_type=DocumentType.KNOWN_ERROR)
    )
    assert known_errors
    assert {m.document_type for m in known_errors} == {DocumentType.KNOWN_ERROR}

    other_service = search_lexical(
        seeded, _QUERY, top_k=10, filters=ChunkFilters(service_name="inventory")
    )
    assert other_service == []


def test_hybrid_is_the_rrf_ranked_union_of_both_arms(seeded: Session) -> None:
    query_vector = FakeEmbeddingProvider(dim=384).embed([_QUERY]).vectors[0]
    vector_hits = search_chunks(seeded, query_vector, top_k=10)
    lexical_hits = search_lexical(seeded, _QUERY, top_k=10)
    hybrid_hits = search_hybrid(seeded, query_embedding=query_vector, query_text=_QUERY, top_k=10)

    union = {m.chunk_id for m in vector_hits} | {m.chunk_id for m in lexical_hits}
    assert {m.chunk_id for m in hybrid_hits} <= union
    assert union  # both arms together found something

    scores = [m.score for m in hybrid_hits]
    assert scores == sorted(scores, reverse=True)

    # the exact-token doc is retrieved by the lexical arm, so it survives fusion
    assert "known-errors/payments/PAY-50231.md" in {m.source_path for m in hybrid_hits}


def test_hybrid_respects_top_k(seeded: Session) -> None:
    query_vector = FakeEmbeddingProvider(dim=384).embed([_QUERY]).vectors[0]
    hits = search_hybrid(
        seeded, query_embedding=query_vector, query_text=_QUERY, top_k=1, candidate_k=10
    )
    assert len(hits) == 1
