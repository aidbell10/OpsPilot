"""Evaluate the Phase 8 agent pipeline and persist it as an ``EvaluationRun``.

Part of Phase 9's "does the agent help?" experiment: the agent's evidence
trail is a heterogeneous sequence of tool observations, not a ranked chunk
list, so it cannot be scored with the deterministic runner's
``recall_at_k``/``mrr``/``ndcg_at_10`` — those keys are simply absent from
this run's metrics (``report.py`` already renders a missing key as "n/a").
What *does* generalize across every pipeline is reused as-is:
``evidence_coverage`` (any evidence text containing a required snippet),
``hallucinated_forbidden_claim``, and the abstention confusion matrix — all
straight from :mod:`opspilot.evaluation.metrics`, unmodified.

Deliberately reuses the ``evaluation_runs``/``evaluation_results`` tables
(``retrieval_strategy=AGENT``) rather than a bespoke report, per the intent
already written into ``report.py``'s docstring back in Phase 1: "agent...
will show up in the same table... with a different retrieval_strategy".
"""

from __future__ import annotations

import time
import uuid

from sqlalchemy.orm import Session

from opspilot.agent.budget import AgentBudget
from opspilot.agent.orchestrator import investigate
from opspilot.config import Settings
from opspilot.evaluation import metrics as m
from opspilot.evaluation.runner import _git_sha, _mean, _percentile, select_cases
from opspilot.models.enums import Difficulty, EvalSplit, RetrievalStrategy
from opspilot.models.evaluation import EvaluationResult, EvaluationRun
from opspilot.providers.base import EmbeddingProvider, LLMProvider, RerankProvider


def run_agent_evaluation(
    session: Session,
    *,
    dataset_version: str,
    split: EvalSplit,
    llm_provider: LLMProvider,
    embedding_provider: EmbeddingProvider,
    rerank_provider: RerankProvider | None,
    settings: Settings,
    difficulties: set[Difficulty] | None = None,
    budget: AgentBudget | None = None,
    notes: str = "",
) -> EvaluationRun:
    """Run the agent over every matching case and persist one ``EvaluationRun``.

    Mirrors :func:`opspilot.evaluation.runner.run_evaluation`'s shape (same
    tables, same aggregate-metric naming where a metric applies) so the two
    show up side by side in ``opspilot.evaluation report`` — see the module
    docstring for which metrics don't carry over and why.
    """
    budget = budget or AgentBudget(
        max_tool_calls=settings.agent_max_tool_calls,
        max_cost_usd=settings.agent_max_cost_usd,
        max_seconds=settings.agent_max_seconds,
    )
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
        retrieval_strategy=RetrievalStrategy.AGENT,
        reranker=rerank_provider.model if rerank_provider is not None else None,
        prompt_version=settings.prompt_version,
        notes=(
            f"{notes} [agent budget: max_tool_calls={budget.max_tool_calls} "
            f"max_cost_usd={budget.max_cost_usd} max_seconds={budget.max_seconds}]"
        ).strip(),
        aggregate_metrics={},
    )
    session.add(run)
    session.flush()

    coverages: list[float] = []
    citation_relevances: list[float] = []
    hallucination_flags: list[bool] = []
    passed_flags: list[bool] = []
    total_latencies: list[float] = []
    tool_calls_used: list[int] = []
    costs_usd: list[float] = []
    abstention_tallies: list[m.AbstentionCounts] = []

    for case in cases:
        t_start = time.perf_counter()
        agent_run = investigate(
            session,
            description=case.query,
            service_name=case.service_name,
            version=None,
            environment=None,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            rerank_provider=rerank_provider,
            settings=settings,
            budget=budget,
        )
        total_ms = (time.perf_counter() - t_start) * 1000
        finding = agent_run.finding

        observation_texts = [o.content for o in agent_run.observations if o.ok]
        evidence_coverage = m.evidence_coverage(observation_texts, case.required_evidence_ids)

        cited = [o for o in agent_run.observations if o.id in finding.citations]
        citation_relevance = (
            m.evidence_coverage([c.content for c in cited], case.required_evidence_ids)
            if cited
            else None
        )

        hallucinated = m.hallucinated_forbidden_claim(
            f"{finding.hypothesis}\n{finding.root_cause or ''}", case.forbidden_claims
        )

        should_abstain = not case.answerable
        did_abstain = not finding.evidence_sufficient
        passed = (
            did_abstain if should_abstain else (finding.evidence_sufficient and not hallucinated)
        )

        result = EvaluationResult(
            id=uuid.uuid4(),
            run_id=run.id,
            case_id=case.case_id,
            retrieval_metrics={
                "evidence_coverage": evidence_coverage,
                "citation_relevance": citation_relevance,
                "recall_at_k": None,
                "mrr": None,
                "ndcg_at_10": None,
                "note": (
                    "agent evidence is a heterogeneous tool trail, not a ranked chunk "
                    "list; recall_at_k/mrr/ndcg_at_10 don't apply"
                ),
            },
            generation_metrics={
                "hallucinated_forbidden_claim": hallucinated,
                "evidence_sufficient": finding.evidence_sufficient,
                "confidence": finding.confidence,
                "n_citations": len(finding.citations),
                "should_abstain": should_abstain,
                "did_abstain": did_abstain,
            },
            latency_ms={"total_ms": total_ms, "tool_calls_used": agent_run.tool_calls_used},
            cost=agent_run.cost,
            raw_output=finding.model_dump(),
            passed=passed,
            total_latency_ms=total_ms,
        )
        session.add(result)

        coverages.append(evidence_coverage)
        if citation_relevance is not None:
            citation_relevances.append(citation_relevance)
        hallucination_flags.append(hallucinated)
        passed_flags.append(passed)
        total_latencies.append(total_ms)
        tool_calls_used.append(agent_run.tool_calls_used)
        estimated_usd = agent_run.cost["estimated_usd"]
        assert isinstance(estimated_usd, float)
        costs_usd.append(estimated_usd)
        abstention_tallies.append(
            m.abstention_counts(should_abstain=should_abstain, did_abstain=did_abstain)
        )

    ab_total = m.sum_abstention_counts(abstention_tallies)

    run.aggregate_metrics = {
        "n_cases": len(cases),
        "mean_evidence_coverage": _mean(coverages),
        "mean_citation_relevance": _mean(citation_relevances) if citation_relevances else None,
        "hallucination_rate": _mean([1.0 if h else 0.0 for h in hallucination_flags]),
        "abstention_precision": ab_total.precision(),
        "abstention_recall": ab_total.recall(),
        "abstention_f1": ab_total.f1(),
        "pass_rate": _mean([1.0 if p else 0.0 for p in passed_flags]),
        "latency_ms_p50": _percentile(total_latencies, 50),
        "latency_ms_p95": _percentile(total_latencies, 95),
        "mean_tool_calls_used": _mean([float(n) for n in tool_calls_used]),
        "total_cost_usd": sum(costs_usd),
        "mean_cost_usd_per_query": _mean(costs_usd),
        "retrieval_strategy": RetrievalStrategy.AGENT.value,
        "reranker": rerank_provider.model if rerank_provider is not None else None,
        "embedding_provider": embedding_provider.name,
        "llm_provider": llm_provider.name,
    }
    session.flush()
    return run
