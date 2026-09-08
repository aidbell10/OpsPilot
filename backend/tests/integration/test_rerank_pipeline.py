"""Phase 6 — the rerank stage inside retrieve_chunks, against a live DB."""

from __future__ import annotations

import pytest
from corpus_fixtures import tiny_corpus
from sqlalchemy.orm import Session

from opspilot.ingestion.pipeline import ingest_corpus
from opspilot.models.enums import RetrievalStrategy
from opspilot.providers.fake import FakeEmbeddingProvider, FakeRerankProvider
from opspilot.retrieval.strategy import retrieve_candidates, retrieve_chunks

pytestmark = [pytest.mark.integration, pytest.mark.retrieval]

_QUERY = "checkout HTTP 500 promotion validate_cart empty discount code"


@pytest.fixture
def seeded(db_session: Session, clean_db: None) -> Session:
    ingest_corpus(
        db_session,
        tiny_corpus(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        chunk_size=48,
        chunk_overlap=8,
    )
    db_session.commit()
    return db_session


def _qv() -> list[float]:
    return FakeEmbeddingProvider(dim=384).embed([_QUERY]).vectors[0]


def test_reranker_trims_candidates_to_top_k(seeded: Session) -> None:
    candidates = retrieve_candidates(
        seeded,
        strategy=RetrievalStrategy.HYBRID_RRF,
        query_embedding=_qv(),
        query_text=_QUERY,
        top_k=10,
    )
    assert len(candidates) >= 3  # tiny_corpus doc chunks into several pieces

    reranked = retrieve_chunks(
        seeded,
        strategy=RetrievalStrategy.HYBRID_RRF,
        query_embedding=_qv(),
        query_text=_QUERY,
        top_k=2,
        reranker=FakeRerankProvider(),
        rerank_candidate_k=10,
    )
    assert len(reranked) == 2
    assert {c.chunk_id for c in reranked} <= {c.chunk_id for c in candidates}
    # score is the reranker's, descending
    assert reranked[0].score >= reranked[1].score


def test_no_reranker_is_a_passthrough(seeded: Session) -> None:
    plain = retrieve_chunks(
        seeded,
        strategy=RetrievalStrategy.HYBRID_RRF,
        query_embedding=_qv(),
        query_text=_QUERY,
        top_k=3,
    )
    same = retrieve_candidates(
        seeded,
        strategy=RetrievalStrategy.HYBRID_RRF,
        query_embedding=_qv(),
        query_text=_QUERY,
        top_k=3,
    )
    assert [c.chunk_id for c in plain] == [c.chunk_id for c in same]
