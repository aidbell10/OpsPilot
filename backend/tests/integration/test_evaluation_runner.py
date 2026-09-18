from __future__ import annotations

import pytest
from corpus_fixtures import tiny_corpus
from eval_fixtures import mixed_difficulty_ground_truth, tiny_ground_truth
from sqlalchemy import select
from sqlalchemy.orm import Session

from opspilot.config import get_settings
from opspilot.evaluation.loader import upsert_ground_truth
from opspilot.evaluation.report import build_report
from opspilot.evaluation.runner import run_evaluation, select_cases
from opspilot.ingestion.pipeline import ingest_corpus
from opspilot.models.enums import Difficulty, EvalSplit, RetrievalStrategy
from opspilot.models.evaluation import EvaluationResult
from opspilot.providers.fake import FakeEmbeddingProvider, FakeLLMProvider, FakeRerankProvider

pytestmark = pytest.mark.integration

_DATASET_VERSION = "test-eval-v1"


def _seed(db_session: Session) -> None:
    provider = FakeEmbeddingProvider(dim=384)
    ingest_corpus(
        db_session, tiny_corpus(), embedding_provider=provider, chunk_size=64, chunk_overlap=8
    )
    upsert_ground_truth(
        db_session, tiny_ground_truth(), dataset_version=_DATASET_VERSION, split=EvalSplit.DEV
    )
    db_session.commit()


def test_run_evaluation_persists_run_and_results(db_session: Session, clean_db: None) -> None:
    _seed(db_session)
    settings = get_settings()

    run = run_evaluation(
        db_session,
        dataset_version=_DATASET_VERSION,
        split=EvalSplit.DEV,
        strategy=RetrievalStrategy.VECTOR,
        embedding_provider=FakeEmbeddingProvider(dim=384),
        llm_provider=FakeLLMProvider(),
        settings=settings,
        notes="integration test",
    )
    db_session.commit()

    assert run.retrieval_strategy is RetrievalStrategy.VECTOR

    assert run.id is not None
    assert run.dataset_version == _DATASET_VERSION
    assert run.aggregate_metrics["n_cases"] == 2

    results = (
        db_session.execute(select(EvaluationResult).where(EvaluationResult.run_id == run.id))
        .scalars()
        .all()
    )
    assert len(results) == 2

    for result in results:
        for key in ("recall_at_k", "evidence_coverage", "mrr", "ndcg_at_10"):
            value = result.retrieval_metrics[key]
            assert 0.0 <= value <= 1.0
        # the fake LLM never emits valid JSON, so every case abstains — a
        # real, honest assertion about the offline provider, not a stub.
        assert result.generation_metrics["evidence_sufficient"] is False
        assert result.generation_metrics["did_abstain"] is True

    by_case = {r.case_id: r for r in results}
    # the answerable case retrieves its one required document...
    assert by_case["gt-checkout-500-01"].retrieval_metrics["recall_at_k"] == 1.0
    assert by_case["gt-checkout-500-01"].retrieval_metrics["evidence_coverage"] == 1.0
    # ...but the fake LLM abstains anyway, so it does not "pass".
    assert by_case["gt-checkout-500-01"].passed is False
    # the unanswerable case has nothing required, and correctly abstained.
    assert by_case["unans-checkout-cpu-01"].passed is True

    metrics = run.aggregate_metrics
    # 1 correctly-abstained unanswerable case (TP) + 1 wrongly-abstained
    # answerable case (FP): precision=1/2, recall=1/1, f1=2/3.
    assert metrics["abstention_precision"] == pytest.approx(0.5)
    assert metrics["abstention_recall"] == pytest.approx(1.0)
    assert metrics["abstention_f1"] == pytest.approx(2 / 3)
    assert metrics["hallucination_rate"] == pytest.approx(0.0)
    # fake LLM never emits citations, so citation precision is undefined for
    # every case and the aggregate must be None, not silently zero.
    assert metrics["mean_citation_precision"] is None
    # fake providers price at $0/token by design (see telemetry/cost.py).
    assert metrics["total_cost_usd"] == pytest.approx(0.0)
    assert metrics["embedding_provider"] == "fake"
    assert metrics["llm_provider"] == "fake"


def test_report_includes_the_persisted_run(db_session: Session, clean_db: None) -> None:
    _seed(db_session)
    settings = get_settings()
    run_evaluation(
        db_session,
        dataset_version=_DATASET_VERSION,
        split=EvalSplit.DEV,
        strategy=RetrievalStrategy.VECTOR,
        embedding_provider=FakeEmbeddingProvider(dim=384),
        llm_provider=FakeLLMProvider(),
        settings=settings,
    )
    db_session.commit()

    report = build_report(db_session)
    assert _DATASET_VERSION in report
    assert "vector" in report


def test_hybrid_and_lexical_strategies_are_recorded(db_session: Session, clean_db: None) -> None:
    _seed(db_session)
    settings = get_settings()

    for strategy in (RetrievalStrategy.LEXICAL, RetrievalStrategy.HYBRID_RRF):
        run = run_evaluation(
            db_session,
            dataset_version=_DATASET_VERSION,
            split=EvalSplit.DEV,
            strategy=strategy,
            embedding_provider=FakeEmbeddingProvider(dim=384),
            llm_provider=FakeLLMProvider(),
            settings=settings,
        )
        db_session.commit()
        assert run.retrieval_strategy is strategy
        assert run.aggregate_metrics["retrieval_strategy"] == strategy.value
        assert run.aggregate_metrics["n_cases"] == 2

    # the answerable case's required doc is lexically obvious ("promotion",
    # "validate_cart", "HTTP 500") — hybrid must still recall it.
    hybrid_results = (
        db_session.execute(
            select(EvaluationResult).where(
                EvaluationResult.case_id == "gt-checkout-500-01",
            )
        )
        .scalars()
        .all()
    )
    assert any(r.retrieval_metrics["recall_at_k"] == 1.0 for r in hybrid_results)

    report = build_report(db_session)
    assert "hybrid_rrf" in report
    assert "lexical" in report


def test_reranker_is_recorded_and_timed(db_session: Session, clean_db: None) -> None:
    _seed(db_session)
    settings = get_settings()

    run = run_evaluation(
        db_session,
        dataset_version=_DATASET_VERSION,
        split=EvalSplit.DEV,
        strategy=RetrievalStrategy.HYBRID_RRF,
        embedding_provider=FakeEmbeddingProvider(dim=384),
        llm_provider=FakeLLMProvider(),
        rerank_provider=FakeRerankProvider(),
        settings=settings,
    )
    db_session.commit()

    assert run.reranker == "fake-rerank-v1"
    assert run.aggregate_metrics["reranker"] == "fake-rerank-v1"

    results = (
        db_session.execute(select(EvaluationResult).where(EvaluationResult.run_id == run.id))
        .scalars()
        .all()
    )
    assert results
    for result in results:
        assert "rerank_ms" in result.latency_ms
        assert result.latency_ms["rerank_ms"] >= 0.0

    assert "fake-rerank-v1" in build_report(db_session)


def test_select_cases_filters_by_difficulty(db_session: Session, clean_db: None) -> None:
    upsert_ground_truth(
        db_session,
        mixed_difficulty_ground_truth(),
        dataset_version=_DATASET_VERSION,
        split=EvalSplit.DEV,
    )
    db_session.commit()

    hard = select_cases(
        db_session,
        dataset_version=_DATASET_VERSION,
        split=EvalSplit.DEV,
        difficulties={Difficulty.MULTI_HOP, Difficulty.ADVERSARIAL},
    )
    assert {c.case_id for c in hard} == {"hard-01", "hard-02"}

    everything = select_cases(
        db_session, dataset_version=_DATASET_VERSION, split=EvalSplit.DEV, difficulties=None
    )
    assert {c.case_id for c in everything} == {"easy-01", "hard-01", "hard-02"}


def test_run_evaluation_respects_the_difficulty_filter(db_session: Session, clean_db: None) -> None:
    ingest_corpus(
        db_session,
        tiny_corpus(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        chunk_size=64,
        chunk_overlap=8,
    )
    upsert_ground_truth(
        db_session,
        mixed_difficulty_ground_truth(),
        dataset_version=_DATASET_VERSION,
        split=EvalSplit.DEV,
    )
    db_session.commit()

    run = run_evaluation(
        db_session,
        dataset_version=_DATASET_VERSION,
        split=EvalSplit.DEV,
        strategy=RetrievalStrategy.VECTOR,
        embedding_provider=FakeEmbeddingProvider(dim=384),
        llm_provider=FakeLLMProvider(),
        settings=get_settings(),
        difficulties={Difficulty.MULTI_HOP, Difficulty.ADVERSARIAL},
    )
    db_session.commit()

    assert run.aggregate_metrics["n_cases"] == 2
    results = (
        db_session.execute(select(EvaluationResult).where(EvaluationResult.run_id == run.id))
        .scalars()
        .all()
    )
    assert {r.case_id for r in results} == {"hard-01", "hard-02"}
