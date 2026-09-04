"""Plain-text comparison report across persisted ``EvaluationRun`` rows.

Renders whatever runs exist — currently only the Phase 3 vector baseline.
Later phases (hybrid retrieval, reranking, fine-tuned reranker, agent) each
add their own ``EvaluationRun`` rows with a different ``retrieval_strategy``/
``reranker``, and will show up in the same table without any change here.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from opspilot.models.evaluation import EvaluationRun

_COLUMNS = [
    "created_at",
    "git_sha",
    "dataset_version",
    "split",
    "strategy",
    "reranker",
    "llm_model",
    "n",
    "pass_rate",
    "recall@k",
    "ndcg@10",
    "abstain_f1",
    "halluc_rate",
    "p50_ms",
    "cost_usd",
]


def _fmt(value: object, spec: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int | float) and spec:
        return format(value, spec)
    return str(value)


def build_report(session: Session) -> str:
    runs = session.execute(select(EvaluationRun).order_by(EvaluationRun.created_at)).scalars().all()
    if not runs:
        return "No evaluation runs recorded yet. Run `uv run python -m opspilot.evaluation run`."

    header = " | ".join(_COLUMNS)
    lines = [header, "-" * len(header)]
    for run in runs:
        metrics = run.aggregate_metrics
        row = [
            run.created_at.isoformat(timespec="seconds"),
            (run.git_sha or "unknown")[:8],
            run.dataset_version,
            run.split.value,
            run.retrieval_strategy.value,
            run.reranker or "none",
            run.llm_model,
            _fmt(metrics.get("n_cases")),
            _fmt(metrics.get("pass_rate"), ".2f"),
            _fmt(metrics.get("mean_recall_at_k"), ".2f"),
            _fmt(metrics.get("mean_ndcg_at_10"), ".2f"),
            _fmt(metrics.get("abstention_f1"), ".2f"),
            _fmt(metrics.get("hallucination_rate"), ".2f"),
            _fmt(metrics.get("latency_ms_p50"), ".0f"),
            _fmt(metrics.get("total_cost_usd"), ".6f"),
        ]
        lines.append(" | ".join(row))
    return "\n".join(lines)
