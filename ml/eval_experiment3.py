"""Experiment 3 — no reranker vs pretrained vs fine-tuned cross-encoder.

Measures Recall@K / MRR / nDCG@10 / rerank latency for three reranker configs
against the 5 evaluation cases in ``ml/datasets/split.json``'s held-out test
chains (chain-01-payments-auth-timeout, chain-06-orders-event-ordering-race) —
chains ``ml/train_reranker.py`` never saw a single pair from. Retrieval
strategy is fixed to hybrid (the project default since Experiment 1).

This intentionally does NOT go through ``opspilot.evaluation.runner`` /
``evaluation_runs``: those tables hold the versioned 22-case dev-split
benchmark (see ``docs/evaluation.md`` on why that dataset isn't chain-split),
and reusing them here would either mix held-out and training-chain queries in
one run or require changing what "dev"/"test" mean for every other
experiment. Retrieval, rerank, and metrics are still 100% the real
production functions (:func:`opspilot.retrieval.strategy.retrieve_candidates`,
:func:`opspilot.retrieval.rerank.rerank_chunks`,
:mod:`opspilot.evaluation.metrics`) — only the case selection and reporting
are bespoke to this held-out slice.

Requires the corpus ingested into the running Postgres (``make ingest``).

Usage: ``uv run python ../ml/eval_experiment3.py`` (run from ``backend/`` —
see ``make ml-experiment3``).
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from opspilot.config import get_settings
from opspilot.db.session import session_scope
from opspilot.evaluation import metrics as m
from opspilot.evaluation.loader import GroundTruthCase, load_ground_truth
from opspilot.models.enums import RetrievalStrategy
from opspilot.providers.base import RerankProvider
from opspilot.providers.factory import get_embedding_provider
from opspilot.retrieval.filters import ChunkFilters
from opspilot.retrieval.rerank import rerank_chunks
from opspilot.retrieval.strategy import retrieve_candidates

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "generated"
ML_DIR = Path(__file__).resolve().parent
SPLIT_PATH = ML_DIR / "datasets" / "split.json"
FINE_TUNED_MODEL = ML_DIR / "checkpoints" / "opspilot-ce-v1"
PRETRAINED_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(pct) - 1]


def _load_test_cases() -> list[GroundTruthCase]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    test_chains = set(split["test_chains"])
    cases = load_ground_truth(DATA_DIR)
    return [c for c in cases if c.chain_id in test_chains]


def _run_config(
    label: str,
    rerank_provider: RerankProvider | None,
    cases: list[GroundTruthCase],
) -> dict[str, object]:
    settings = get_settings()
    embedding_provider = get_embedding_provider()

    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    rerank_latencies: list[float] = []

    with session_scope() as session:
        for case in cases:
            query_vector = embedding_provider.embed([case.query]).vectors[0]
            fetch_k = (
                max(settings.retrieval_top_k, settings.rerank_candidate_k)
                if rerank_provider is not None
                else settings.retrieval_top_k
            )
            candidates = retrieve_candidates(
                session,
                strategy=RetrievalStrategy.HYBRID_RRF,
                query_embedding=query_vector,
                query_text=case.query,
                top_k=fetch_k,
                filters=ChunkFilters(service_name=case.service_name),
                candidate_k=settings.retrieval_candidate_k,
                rrf_k=settings.rrf_k,
            )
            if rerank_provider is not None:
                t0 = time.perf_counter()
                chunks = rerank_chunks(
                    rerank_provider, case.query, candidates, top_k=settings.retrieval_top_k
                )
                rerank_latencies.append((time.perf_counter() - t0) * 1000)
            else:
                chunks = candidates
                rerank_latencies.append(0.0)

            retrieved_paths = [c.source_path for c in chunks if c.source_path]
            required_paths = case.required_document_source_paths
            recalls.append(m.recall_at_k(retrieved_paths, required_paths))
            mrrs.append(m.mrr(retrieved_paths, required_paths))
            ndcgs.append(m.ndcg_at_10(retrieved_paths, required_paths))

    return {
        "config": label,
        "n_cases": len(cases),
        "mean_recall_at_k": sum(recalls) / len(recalls),
        "mean_mrr": sum(mrrs) / len(mrrs),
        "mean_ndcg_at_10": sum(ndcgs) / len(ndcgs),
        "rerank_ms_p50": _percentile(rerank_latencies, 50),
        "rerank_ms_p95": _percentile(rerank_latencies, 95),
    }


def main() -> int:
    cases = _load_test_cases()
    if not cases:
        print("No held-out test cases found — run `make ml-dataset` first.")
        return 1

    from opspilot.providers.local_rerank import LocalCrossEncoderProvider

    if not FINE_TUNED_MODEL.exists():
        print(f"No fine-tuned checkpoint at {FINE_TUNED_MODEL} — run `make ml-train` first.")
        return 1

    configs: list[tuple[str, RerankProvider | None]] = [
        ("A. hybrid, no reranker", None),
        ("B. hybrid + pretrained cross-encoder", LocalCrossEncoderProvider(model=PRETRAINED_MODEL)),
        (
            "C. hybrid + fine-tuned cross-encoder",
            LocalCrossEncoderProvider(model=str(FINE_TUNED_MODEL)),
        ),
    ]

    results = [_run_config(label, provider, cases) for label, provider in configs]

    header = f"{'config':<40} {'Recall@K':>9} {'MRR':>7} {'nDCG@10':>8} {'p50 ms':>8} {'p95 ms':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['config']:<40} {r['mean_recall_at_k']:>9.3f} {r['mean_mrr']:>7.3f} "
            f"{r['mean_ndcg_at_10']:>8.3f} {r['rerank_ms_p50']:>8.1f} {r['rerank_ms_p95']:>8.1f}"
        )

    report_dir = ML_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "experiment3_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
