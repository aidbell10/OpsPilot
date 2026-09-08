from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from corpus_fixtures import tiny_corpus
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from opspilot.db.session import get_db
from opspilot.ingestion.pipeline import ingest_corpus
from opspilot.main import create_app
from opspilot.providers.fake import FakeEmbeddingProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    app = create_app()

    def _get_db() -> Iterator[Session]:
        yield db_session

    app.dependency_overrides[get_db] = _get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_search_with_no_data_returns_empty_results(client: TestClient, clean_db: None) -> None:
    resp = client.post("/search", json={"query": "checkout 500 errors"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "checkout 500 errors"
    assert body["strategy"] == "hybrid_rrf"  # default OPSPILOT_RETRIEVAL_STRATEGY
    assert body["results"] == []


def test_search_strategy_override_is_echoed(
    client: TestClient, db_session: Session, clean_db: None
) -> None:
    ingest_corpus(
        db_session,
        tiny_corpus(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        chunk_size=64,
        chunk_overlap=8,
    )
    db_session.commit()

    resp = client.post(
        "/search",
        json={"query": "checkout HTTP 500 promotion validate_cart", "strategy": "lexical"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "lexical"
    assert body["reranker"] is None
    assert len(body["results"]) >= 1


def test_search_with_reranker_enabled(
    client: TestClient,
    db_session: Session,
    clean_db: None,
    set_env: Callable[..., None],
) -> None:
    from opspilot.providers.factory import get_rerank_provider

    ingest_corpus(
        db_session,
        tiny_corpus(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        chunk_size=48,
        chunk_overlap=8,
    )
    db_session.commit()

    set_env(reranker="fake")
    get_rerank_provider.cache_clear()

    resp = client.post("/search", json={"query": "checkout HTTP 500 promotion", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reranker"] == "fake-rerank-v1"
    assert 1 <= len(body["results"]) <= 3

    # rerank=false forces it off even when configured
    resp_off = client.post(
        "/search", json={"query": "checkout HTTP 500 promotion", "rerank": False}
    )
    assert resp_off.json()["reranker"] is None


def test_search_returns_ingested_evidence(
    client: TestClient, db_session: Session, clean_db: None
) -> None:
    ingest_corpus(
        db_session,
        tiny_corpus(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        chunk_size=64,
        chunk_overlap=8,
    )
    db_session.commit()

    resp = client.post("/search", json={"query": "checkout HTTP 500 promotion", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) >= 1
    assert body["results"][0]["service_name"] == "checkout"


def test_search_document_type_filter(
    client: TestClient, db_session: Session, clean_db: None
) -> None:
    ingest_corpus(
        db_session,
        tiny_corpus(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        chunk_size=64,
        chunk_overlap=8,
    )
    db_session.commit()

    resp = client.post(
        "/search",
        json={"query": "checkout HTTP 500", "document_type": "postmortem"},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_analyze_with_fake_llm_returns_abstained_result(client: TestClient, clean_db: None) -> None:
    resp = client.post(
        "/incidents/analyze",
        json={"description": "checkout is returning HTTP 500 after v2.14.0 deploy"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["analysis"]["evidence_sufficient"] is False
    assert body["analysis"]["abstain_reason"]
    assert body["incident_id"]
    assert body["retrieval_strategy"] == "hybrid_rrf"
    assert body["reranker"] is None  # OPSPILOT_RERANKER=none by default


def test_analyze_persists_an_incident_row(
    client: TestClient, db_session: Session, clean_db: None
) -> None:
    from sqlalchemy import select

    from opspilot.models.incident import Incident

    resp = client.post(
        "/incidents/analyze",
        json={"description": "checkout is returning HTTP 500 after v2.14.0 deploy"},
    )
    incident_id = resp.json()["incident_id"]
    stored = db_session.execute(
        select(Incident).where(Incident.id == incident_id)
    ).scalar_one_or_none()
    assert stored is not None
    assert stored.analysis is not None


def test_feedback_is_persisted(client: TestClient, clean_db: None) -> None:
    resp = client.post("/feedback", json={"comment": "helpful", "rating": 5, "helpful": True})
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"]
    assert body["created_at"]
