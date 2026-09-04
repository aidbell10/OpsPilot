"""CLI entry point.

uv run python -m opspilot.ingestion                  # ingest data/generated/ into the configured DB
uv run python -m opspilot.ingestion --data-dir path
uv run python -m opspilot.ingestion --dry-run         # load + chunk only; write nothing
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opspilot.config import get_settings
from opspilot.db.session import session_scope
from opspilot.ingestion.chunking import chunk_text
from opspilot.ingestion.loader import load_corpus
from opspilot.ingestion.pipeline import ingest_corpus
from opspilot.providers.factory import get_embedding_provider

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "generated"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="opspilot.ingestion", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--dry-run", action="store_true", help="load + chunk only; skip embeddings and DB writes"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus = load_corpus(args.data_dir)
    settings = get_settings()

    if args.dry_run:
        n_chunks = sum(
            len(
                chunk_text(
                    d.content, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
                )
            )
            for d in corpus.documents
        )
        print(
            f"dry-run ({args.data_dir}): {len(corpus.services)} services, "
            f"{len(corpus.documents)} documents ({n_chunks} chunks), "
            f"{len(corpus.deployments)} deployments, "
            f"{len(corpus.historical_incidents)} historical incidents"
        )
        return 0

    embedding_provider = get_embedding_provider()
    with session_scope() as session:
        summary = ingest_corpus(
            session,
            corpus,
            embedding_provider=embedding_provider,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    print(
        f"ingested: {summary.services} services, {summary.documents} documents "
        f"({summary.chunks} chunks), {summary.deployments} deployments, "
        f"{summary.historical_incidents} historical incidents "
        f"[embedding_provider={embedding_provider.name} model={embedding_provider.model}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
