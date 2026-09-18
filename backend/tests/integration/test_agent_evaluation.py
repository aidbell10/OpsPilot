"""Phase 9 — the agent pipeline through the evaluation harness (``--strategy agent``)."""

from __future__ import annotations

import json

import pytest
from corpus_fixtures import agent_corpus
from eval_fixtures import mixed_difficulty_ground_truth, tiny_ground_truth
from sqlalchemy import select
from sqlalchemy.orm import Session

from opspilot.config import get_settings
from opspilot.evaluation.agent_runner import run_agent_evaluation
from opspilot.evaluation.loader import GroundTruthCase, upsert_ground_truth
from opspilot.evaluation.report import build_report
from opspilot.ingestion.pipeline import ingest_corpus
from opspilot.models.enums import Difficulty, EvalSplit, RetrievalStrategy
from opspilot.models.evaluation import EvaluationResult
from opspilot.providers.base import LLMResult
from opspilot.providers.fake import FakeEmbeddingProvider, FakeLLMProvider

pytestmark = [pytest.mark.integration, pytest.mark.agent]

_DATASET_VERSION = "test-agent-eval-v1"


class _SearchThenAnswerLLM:
    """Calls search_docs once, then answers citing obs-1 — for any case's query."""

    name = "scripted"
    model = "scripted-v1"

    def complete(
        self, *, system: str, user: str, max_tokens: int, temperature: float = 0.0
    ) -> LLMResult:
        if "this is your first turn" in user:
            query = user.split("## Reported incident\n\n", 1)[1].split("\n\n", 1)[0]
            text = json.dumps(
                {"action": "call_tool", "tool": "search_docs", "arguments": {"query": query}}
            )
        else:
            text = json.dumps(
                {
                    "action": "final_answer",
                    "hypothesis": "promotion validation bug",
                    "confidence": 0.9,
                    "evidence_sufficient": True,
                    "citations": ["obs-1"],
                }
            )
        return LLMResult(text=text, model=self.model, input_tokens=5, output_tokens=5)


def _seed(db_session: Session, cases: list[GroundTruthCase]) -> None:
    ingest_corpus(
        db_session,
        agent_corpus(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        chunk_size=48,
        chunk_overlap=8,
    )
    upsert_ground_truth(db_session, cases, dataset_version=_DATASET_VERSION, split=EvalSplit.DEV)
    db_session.commit()


def test_run_agent_evaluation_persists_a_run_with_agent_strategy(
    db_session: Session, clean_db: None
) -> None:
    _seed(db_session, tiny_ground_truth())
    settings = get_settings()

    run = run_agent_evaluation(
        db_session,
        dataset_version=_DATASET_VERSION,
        split=EvalSplit.DEV,
        llm_provider=_SearchThenAnswerLLM(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        rerank_provider=None,
        settings=settings,
        notes="unit test",
    )
    db_session.commit()

    assert run.retrieval_strategy is RetrievalStrategy.AGENT
    assert run.aggregate_metrics["n_cases"] == 2
    assert "agent budget" in run.notes

    results = (
        db_session.execute(select(EvaluationResult).where(EvaluationResult.run_id == run.id))
        .scalars()
        .all()
    )
    assert len(results) == 2
    for result in results:
        # recall/mrr/ndcg are explicitly not-applicable for the agent pipeline.
        assert result.retrieval_metrics["recall_at_k"] is None
        assert result.retrieval_metrics["mrr"] is None
        assert result.retrieval_metrics["ndcg_at_10"] is None
        assert 0.0 <= result.retrieval_metrics["evidence_coverage"] <= 1.0
        assert "tool_calls_used" in result.latency_ms

    by_case = {r.case_id: r for r in results}
    # the answerable case's runbook is findable by search_docs and cited.
    assert by_case["gt-checkout-500-01"].retrieval_metrics["evidence_coverage"] == 1.0
    assert by_case["gt-checkout-500-01"].passed is True
    # the unanswerable case has no evidence to find; the script still answers
    # "sufficient" (it's scripted, not smart) — a real LLM might abstain
    # instead, which is exactly what Experiment 4 investigates.
    assert by_case["unans-checkout-cpu-01"].generation_metrics["did_abstain"] is False

    metrics = run.aggregate_metrics
    for key in (
        "mean_evidence_coverage",
        "hallucination_rate",
        "abstention_precision",
        "abstention_recall",
        "abstention_f1",
        "pass_rate",
        "latency_ms_p50",
        "mean_tool_calls_used",
        "total_cost_usd",
    ):
        assert key in metrics
    assert metrics["mean_tool_calls_used"] == pytest.approx(1.0)
    assert metrics["retrieval_strategy"] == "agent"


def test_run_agent_evaluation_respects_the_difficulty_filter(
    db_session: Session, clean_db: None
) -> None:
    _seed(db_session, mixed_difficulty_ground_truth())
    settings = get_settings()

    run = run_agent_evaluation(
        db_session,
        dataset_version=_DATASET_VERSION,
        split=EvalSplit.DEV,
        llm_provider=_SearchThenAnswerLLM(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        rerank_provider=None,
        settings=settings,
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


def test_agent_run_with_fake_provider_abstains_with_zero_tool_calls(
    db_session: Session, clean_db: None
) -> None:
    _seed(db_session, tiny_ground_truth())
    settings = get_settings()

    run = run_agent_evaluation(
        db_session,
        dataset_version=_DATASET_VERSION,
        split=EvalSplit.DEV,
        llm_provider=FakeLLMProvider(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        rerank_provider=None,
        settings=settings,
    )
    db_session.commit()

    assert run.aggregate_metrics["mean_tool_calls_used"] == pytest.approx(0.0)
    assert run.aggregate_metrics["abstention_recall"] == pytest.approx(1.0)


def test_report_includes_the_agent_run_and_handles_missing_retrieval_metrics(
    db_session: Session, clean_db: None
) -> None:
    _seed(db_session, tiny_ground_truth())
    settings = get_settings()

    run_agent_evaluation(
        db_session,
        dataset_version=_DATASET_VERSION,
        split=EvalSplit.DEV,
        llm_provider=_SearchThenAnswerLLM(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        rerank_provider=None,
        settings=settings,
    )
    db_session.commit()

    report = build_report(db_session)
    assert "agent" in report
    assert "n/a" in report  # recall@k / ndcg@10 are not applicable to the agent row
