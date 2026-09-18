"""Run the RAG pipeline over a set of evaluation cases, for a given strategy.

Reuses the exact retrieval and generation-parsing functions the API uses
(:func:`opspilot.retrieval.strategy.retrieve_candidates` +
:func:`opspilot.retrieval.rerank.rerank_chunks` — together these are what
:func:`opspilot.retrieval.strategy.retrieve_chunks` runs for
``POST /search`` and ``POST /incidents/analyze`` —
:func:`opspilot.retrieval.semantic.search_historical_incidents`,
:func:`opspilot.generation.prompt.build_user_prompt`,
:func:`opspilot.generation.parser.parse_llm_output`) — the retrieval,
rerank, and generation *logic* is never reimplemented here.

Two deliberate deviations from the endpoints, both to capture telemetry the
endpoint wrappers hide: (1) the query is embedded once here (reused for the
chunk and historical-incident searches) and the LLM provider is called
directly, rather than via ``run_generation`` / the ``embed_and_search_*``
wrappers, so the embedding/LLM token counts reach cost accounting; (2)
retrieval and rerank are called as separate timed steps rather than as the
single ``retrieve_chunks`` call, so ``retrieval_ms`` and ``rerank_ms`` land in
``latency_ms`` independently. The queries and the parse/validate/citation-verify
logic are 100% shared.
"""

from __future__ import annotations

import datetime as dt
import statistics
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from opspilot.config import Settings
from opspilot.evaluation import metrics as m
from opspilot.generation.parser import parse_llm_output
from opspilot.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from opspilot.models.enums import Difficulty, EvalSplit, RetrievalStrategy
from opspilot.models.evaluation import EvaluationCase, EvaluationResult, EvaluationRun
from opspilot.providers.base import EmbeddingProvider, LLMProvider, RerankProvider
from opspilot.retrieval.filters import ChunkFilters
from opspilot.retrieval.rerank import rerank_chunks
from opspilot.retrieval.semantic import search_historical_incidents
from opspilot.retrieval.strategy import retrieve_candidates
from opspilot.schemas.analysis import RelatedIncident
from opspilot.telemetry.cost import CostAccumulator

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(pct) - 1]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    retrieval_metrics: dict[str, object]
    generation_metrics: dict[str, object]
    latency_ms: dict[str, float]
    cost: dict[str, float | str]
    raw_output: dict[str, object]
    passed: bool


def _evaluate_case(
    session: Session,
    case: EvaluationCase,
    *,
    strategy: RetrievalStrategy,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    rerank_provider: RerankProvider | None,
    settings: Settings,
) -> CaseEvaluation:
    """Run retrieval + generation for one case; return the JSONB payloads to persist."""
    t_start = time.perf_counter()

    embed_result = embedding_provider.embed([case.query])
    query_vector = embed_result.vectors[0]

    fetch_k = (
        max(settings.retrieval_top_k, settings.rerank_candidate_k)
        if rerank_provider is not None
        else settings.retrieval_top_k
    )
    t_retrieval_start = time.perf_counter()
    candidates = retrieve_candidates(
        session,
        strategy=strategy,
        query_embedding=query_vector,
        query_text=case.query,
        top_k=fetch_k,
        filters=ChunkFilters(service_name=case.service_name),
        candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.rrf_k,
    )
    historical_matches = search_historical_incidents(
        session,
        query_vector,
        top_k=min(5, settings.retrieval_top_k),
        service_name=case.service_name,
    )
    retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000

    if rerank_provider is not None:
        t_rerank_start = time.perf_counter()
        chunk_matches = rerank_chunks(
            rerank_provider, case.query, candidates, top_k=settings.retrieval_top_k
        )
        rerank_ms = (time.perf_counter() - t_rerank_start) * 1000
    else:
        chunk_matches = candidates
        rerank_ms = 0.0

    related_incidents = [
        RelatedIncident(
            incident_id=str(hm.incident_id),
            title=hm.title,
            service_name=hm.service_name,
            occurred_at=dt.datetime.fromisoformat(hm.occurred_at),
            severity=hm.severity,
            root_cause=hm.root_cause,
            resolution=hm.resolution,
            score=hm.score,
        )
        for hm in historical_matches
    ]

    user_prompt = build_user_prompt(
        description=case.query,
        service=case.service_name,
        version=None,
        environment=None,
        evidence=chunk_matches,
        related_incidents=related_incidents,
    )

    t_generation_start = time.perf_counter()
    completion = llm_provider.complete(
        system=SYSTEM_PROMPT, user=user_prompt, max_tokens=settings.llm_max_tokens
    )
    valid_chunk_ids = {str(cm.chunk_id) for cm in chunk_matches}
    analysis = parse_llm_output(completion.text, valid_chunk_ids=valid_chunk_ids)
    generation_ms = (time.perf_counter() - t_generation_start) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000

    retrieved_paths_ordered = [cm.source_path for cm in chunk_matches if cm.source_path]
    retrieved_contents = [cm.content for cm in chunk_matches]
    chunk_id_to_path = {str(cm.chunk_id): cm.source_path for cm in chunk_matches if cm.source_path}
    cited_paths = [chunk_id_to_path[cid] for cid in analysis.citations if cid in chunk_id_to_path]

    retrieval_metrics: dict[str, object] = {
        "recall_at_k": m.recall_at_k(retrieved_paths_ordered, case.required_document_ids),
        "evidence_coverage": m.evidence_coverage(retrieved_contents, case.required_evidence_ids),
        "mrr": m.mrr(retrieved_paths_ordered, case.required_document_ids),
        "ndcg_at_10": m.ndcg_at_10(retrieved_paths_ordered, case.required_document_ids),
    }

    hallucinated = m.hallucinated_forbidden_claim(
        f"{analysis.hypothesis}\n{analysis.root_cause or ''}", case.forbidden_claims
    )
    generation_metrics: dict[str, object] = {
        "citation_precision": m.citation_precision(cited_paths, case.required_document_ids),
        "hallucinated_forbidden_claim": hallucinated,
        "evidence_sufficient": analysis.evidence_sufficient,
        "confidence": analysis.confidence,
        "n_citations": len(analysis.citations),
    }

    latency_ms: dict[str, float] = {
        "retrieval_ms": retrieval_ms,
        "rerank_ms": rerank_ms,
        "generation_ms": generation_ms,
        "total_ms": total_ms,
    }

    cost_acc = CostAccumulator()
    cost_acc.add_embedding(embedding_provider.model, embed_result.total_tokens)
    cost_acc.add_llm(llm_provider.model, completion.input_tokens, completion.output_tokens)
    cost = cost_acc.as_dict()

    should_abstain = not case.answerable
    did_abstain = not analysis.evidence_sufficient
    if case.answerable:
        required_recall = retrieval_metrics["recall_at_k"]
        assert isinstance(required_recall, float)
        passed = required_recall >= 1.0 and analysis.evidence_sufficient and not hallucinated
    else:
        passed = did_abstain

    generation_metrics["should_abstain"] = should_abstain
    generation_metrics["did_abstain"] = did_abstain

    return CaseEvaluation(
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        latency_ms=latency_ms,
        cost=cost,
        raw_output=analysis.model_dump(),
        passed=passed,
    )


def select_cases(
    session: Session,
    *,
    dataset_version: str,
    split: EvalSplit,
    difficulties: set[Difficulty] | None = None,
) -> list[EvaluationCase]:
    """The cases a run should evaluate — shared by the deterministic and agent runners.

    ``difficulties`` restricts to a subset (e.g. ``{MULTI_HOP, ADVERSARIAL}`` for
    Phase 9's "hard incidents" experiment); ``None`` means every case in the
    split, matching every runner's behavior before Phase 9.
    """
    stmt = select(EvaluationCase).where(
        EvaluationCase.dataset_version == dataset_version,
        EvaluationCase.split == split,
    )
    if difficulties:
        stmt = stmt.where(EvaluationCase.difficulty.in_(difficulties))
    return list(session.execute(stmt).scalars().all())


def run_evaluation(
    session: Session,
    *,
    dataset_version: str,
    split: EvalSplit,
    strategy: RetrievalStrategy,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    rerank_provider: RerankProvider | None = None,
    settings: Settings,
    difficulties: set[Difficulty] | None = None,
    notes: str = "",
) -> EvaluationRun:
    """Evaluate every matching case in ``dataset_version``/``split`` and persist one run.

    Each call creates a brand-new ``EvaluationRun`` (evaluation runs are an
    append-only history, not idempotent by natural key — re-running is how you
    compare a system change against the past, so each run must be its own row).
    Caller is responsible for committing.
    """
    cases = select_cases(
        session, dataset_version=dataset_version, split=split, difficulties=difficulties
    )

    run = EvaluationRun(
        id=uuid.uuid4(),
        git_sha=_git_sha(),
        dataset_version=dataset_version,
        split=split,
        llm_model=llm_provider.model,
        embedding_model=embedding_provider.model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.retrieval_top_k,
        retrieval_strategy=strategy,
        reranker=rerank_provider.model if rerank_provider is not None else None,
        prompt_version=settings.prompt_version,
        notes=notes,
        aggregate_metrics={},
    )
    session.add(run)
    session.flush()

    recalls: list[float] = []
    coverages: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    citation_precisions: list[float] = []
    hallucination_flags: list[bool] = []
    passed_flags: list[bool] = []
    total_latencies: list[float] = []
    retrieval_latencies: list[float] = []
    rerank_latencies: list[float] = []
    costs_usd: list[float] = []
    abstention_tallies: list[m.AbstentionCounts] = []

    for case in cases:
        evaluated = _evaluate_case(
            session,
            case,
            strategy=strategy,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            rerank_provider=rerank_provider,
            settings=settings,
        )

        result = EvaluationResult(
            id=uuid.uuid4(),
            run_id=run.id,
            case_id=case.case_id,
            retrieval_metrics=evaluated.retrieval_metrics,
            generation_metrics=evaluated.generation_metrics,
            latency_ms=evaluated.latency_ms,
            cost=evaluated.cost,
            raw_output=evaluated.raw_output,
            passed=evaluated.passed,
            total_latency_ms=evaluated.latency_ms["total_ms"],
        )
        session.add(result)

        retrieval_metrics = evaluated.retrieval_metrics
        generation_metrics = evaluated.generation_metrics
        recalls.append(float(retrieval_metrics["recall_at_k"]))  # type: ignore[arg-type]
        coverages.append(float(retrieval_metrics["evidence_coverage"]))  # type: ignore[arg-type]
        mrrs.append(float(retrieval_metrics["mrr"]))  # type: ignore[arg-type]
        ndcgs.append(float(retrieval_metrics["ndcg_at_10"]))  # type: ignore[arg-type]
        cp = generation_metrics["citation_precision"]
        if cp is not None:
            citation_precisions.append(float(cp))  # type: ignore[arg-type]
        hallucination_flags.append(bool(generation_metrics["hallucinated_forbidden_claim"]))
        passed_flags.append(evaluated.passed)
        total_latencies.append(evaluated.latency_ms["total_ms"])
        retrieval_latencies.append(evaluated.latency_ms["retrieval_ms"])
        rerank_latencies.append(evaluated.latency_ms["rerank_ms"])
        estimated_usd = evaluated.cost["estimated_usd"]
        assert isinstance(estimated_usd, float)
        costs_usd.append(estimated_usd)
        abstention_tallies.append(
            m.abstention_counts(
                should_abstain=bool(generation_metrics["should_abstain"]),
                did_abstain=bool(generation_metrics["did_abstain"]),
            )
        )

    ab_total = m.sum_abstention_counts(abstention_tallies)

    run.aggregate_metrics = {
        "n_cases": len(cases),
        "mean_recall_at_k": _mean(recalls),
        "mean_evidence_coverage": _mean(coverages),
        "mean_mrr": _mean(mrrs),
        "mean_ndcg_at_10": _mean(ndcgs),
        "mean_citation_precision": _mean(citation_precisions) if citation_precisions else None,
        "hallucination_rate": _mean([1.0 if h else 0.0 for h in hallucination_flags]),
        "abstention_precision": ab_total.precision(),
        "abstention_recall": ab_total.recall(),
        "abstention_f1": ab_total.f1(),
        "pass_rate": _mean([1.0 if p else 0.0 for p in passed_flags]),
        "latency_ms_p50": _percentile(total_latencies, 50),
        "latency_ms_p95": _percentile(total_latencies, 95),
        "mean_retrieval_ms": _mean(retrieval_latencies),
        "mean_rerank_ms": _mean(rerank_latencies),
        "total_cost_usd": sum(costs_usd),
        "mean_cost_usd_per_query": _mean(costs_usd),
        "retrieval_strategy": strategy.value,
        "reranker": rerank_provider.model if rerank_provider is not None else None,
        "embedding_provider": embedding_provider.name,
        "llm_provider": llm_provider.name,
    }
    session.flush()
    return run
