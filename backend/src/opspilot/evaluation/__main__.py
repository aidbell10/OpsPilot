"""CLI entry point.

uv run python -m opspilot.evaluation load-cases [--data-dir path] [--dataset-version v] [--dry-run]
uv run python -m opspilot.evaluation run [--dataset-version v] [--split dev|test]
    [--strategy vector|lexical|hybrid|agent] [--reranker none|fake|cross_encoder]
    [--difficulty straightforward,multi_hop,unanswerable,adversarial]
    [--agent-max-tool-calls N] [--notes "..."]
uv run python -m opspilot.evaluation report
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opspilot.agent.budget import AgentBudget
from opspilot.config import get_settings
from opspilot.db.session import session_scope
from opspilot.evaluation.agent_runner import run_agent_evaluation
from opspilot.evaluation.loader import (
    dataset_version_from_manifest,
    load_ground_truth,
    upsert_ground_truth,
)
from opspilot.evaluation.report import build_report
from opspilot.evaluation.runner import run_evaluation
from opspilot.models.enums import Difficulty, EvalSplit, RetrievalStrategy
from opspilot.providers.factory import (
    build_rerank_provider,
    get_embedding_provider,
    get_llm_provider,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "generated"


def _cmd_load_cases(args: argparse.Namespace) -> int:
    cases = load_ground_truth(args.data_dir)
    dataset_version = args.dataset_version or dataset_version_from_manifest(args.data_dir)

    if args.dry_run:
        answerable = sum(1 for c in cases if c.answerable)
        print(
            f"dry-run ({args.data_dir}): {len(cases)} ground-truth cases "
            f"({answerable} answerable, {len(cases) - answerable} unanswerable) "
            f"[dataset_version={dataset_version}]"
        )
        return 0

    with session_scope() as session:
        count = upsert_ground_truth(
            session, cases, dataset_version=dataset_version, split=EvalSplit.DEV
        )
    print(f"loaded {count} evaluation cases [dataset_version={dataset_version} split=dev]")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    dataset_version = args.dataset_version or dataset_version_from_manifest(DEFAULT_DATA_DIR)
    split = EvalSplit(args.split)
    strategy = RetrievalStrategy.from_name(args.strategy or settings.retrieval_strategy)
    rerank_provider = build_rerank_provider(args.reranker or settings.reranker, settings)
    difficulties = (
        {Difficulty(d.strip()) for d in args.difficulty.split(",")} if args.difficulty else None
    )

    with session_scope() as session:
        if strategy is RetrievalStrategy.AGENT:
            budget = AgentBudget(
                max_tool_calls=args.agent_max_tool_calls or settings.agent_max_tool_calls,
                max_cost_usd=settings.agent_max_cost_usd,
                max_seconds=settings.agent_max_seconds,
            )
            run = run_agent_evaluation(
                session,
                dataset_version=dataset_version,
                split=split,
                llm_provider=get_llm_provider(),
                embedding_provider=get_embedding_provider(),
                rerank_provider=rerank_provider,
                settings=settings,
                difficulties=difficulties,
                budget=budget,
                notes=args.notes,
            )
        else:
            run = run_evaluation(
                session,
                dataset_version=dataset_version,
                split=split,
                strategy=strategy,
                embedding_provider=get_embedding_provider(),
                llm_provider=get_llm_provider(),
                rerank_provider=rerank_provider,
                settings=settings,
                difficulties=difficulties,
                notes=args.notes,
            )
        print(
            f"run {run.id} [dataset_version={dataset_version} split={split.value} "
            f"strategy={run.retrieval_strategy.value} reranker={run.reranker or 'none'} "
            f"git_sha={run.git_sha} llm={run.llm_model} embedding={run.embedding_model}]"
        )
        for key, value in run.aggregate_metrics.items():
            print(f"  {key}: {value}")
    return 0


def _cmd_report(_args: argparse.Namespace) -> int:
    with session_scope() as session:
        print(build_report(session))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="opspilot.evaluation", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load-cases", help="load ground_truth.json into evaluation_cases")
    p_load.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_load.add_argument("--dataset-version", type=str, default=None)
    p_load.add_argument("--dry-run", action="store_true")
    p_load.set_defaults(func=_cmd_load_cases)

    p_run = sub.add_parser("run", help="run the baseline pipeline over a loaded dataset/split")
    p_run.add_argument("--dataset-version", type=str, default=None)
    p_run.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    p_run.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["vector", "lexical", "hybrid", "agent"],
        help=(
            "pipeline for this run (default: OPSPILOT_RETRIEVAL_STRATEGY). "
            "'agent' runs the Phase 8 LangGraph agent instead of the deterministic "
            "retrieve-then-generate pipeline (see opspilot.evaluation.agent_runner); "
            "--reranker still applies to the agent's own search_docs tool."
        ),
    )
    p_run.add_argument(
        "--reranker",
        type=str,
        default=None,
        choices=["none", "fake", "cross_encoder"],
        help="cross-encoder reranker for this run (default: OPSPILOT_RERANKER)",
    )
    p_run.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help=(
            "comma-separated subset of straightforward,multi_hop,unanswerable,adversarial "
            "(default: every difficulty in the split) — e.g. multi_hop,adversarial for "
            "Phase 9's 'hard incidents' experiment"
        ),
    )
    p_run.add_argument(
        "--agent-max-tool-calls",
        type=int,
        default=None,
        help="override OPSPILOT_AGENT_MAX_TOOL_CALLS for an --strategy agent run",
    )
    p_run.add_argument("--notes", type=str, default="")
    p_run.set_defaults(func=_cmd_run)

    p_report = sub.add_parser("report", help="render a comparison table across all runs")
    p_report.set_defaults(func=_cmd_report)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
