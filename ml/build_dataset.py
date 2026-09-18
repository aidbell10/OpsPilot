"""Build a labeled (query, passage, label) dataset for cross-encoder fine-tuning.

Source: the 12 planted ground-truth chains in ``data/generated`` (Phase 2).
For each answerable evaluation case (18 of the 22 — the other 4 are
unanswerable and carry no required evidence), every document named in
``required_document_source_paths`` is chunked exactly as ingestion chunks it
(:func:`opspilot.ingestion.chunking.chunk_text`, same ``chunk_size``/
``chunk_overlap`` as the running config) and the chunk containing one of the
case's ``required_evidence_snippets`` becomes a *positive* pair. Other
documents belonging to the same service are sampled as *hard negatives*
(same service, wrong chain — the confusion a reranker actually needs to
learn to resolve) and a couple of documents from unrelated services as *easy
negatives* (regularization; a pretrained cross-encoder already handles these).

Split is by chain_id, not by case, so no chunk or near-duplicate query ever
crosses a split boundary:

  train: 8 chains / 9 cases  (fine-tuning signal)
  dev:   2 chains / 4 cases  (used only to pick the best training checkpoint)
  test:  2 chains / 5 cases  (held out of fine-tuning entirely — Experiment 3
                              measures Recall@K/MRR/nDCG@10 on these chains only,
                              via ml/eval_experiment3.py)

The chain assignment is a fixed, hand-picked list rather than a random split:
with only 12 chains, a random draw risks a 1-chain test set (too small for a
non-degenerate Recall@K); the choice below is documented, not tuned against
the result.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from opspilot.config import get_settings
from opspilot.evaluation.loader import GroundTruthCase, load_ground_truth
from opspilot.ingestion.chunking import Chunk, chunk_text
from opspilot.ingestion.loader import Corpus, DocumentRecord, load_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "generated"
OUT_DIR = Path(__file__).resolve().parent / "datasets"

SEED = 42
HARD_NEGATIVES_PER_CASE = 5
EASY_NEGATIVES_PER_CASE = 2

TEST_CHAINS = {"chain-01-payments-auth-timeout", "chain-06-orders-event-ordering-race"}
DEV_CHAINS = {"chain-02-auth-jwks-stale", "chain-09-inventory-kafka-rebalance"}
SPLIT_RATIONALE = (
    "Hand-picked chain-level split (not random): with only 12 planted chains, a "
    "random draw risks a near-empty test set. test = chain-01 + chain-06 (5 cases, "
    "both multi_hop, payments + orders); dev = chain-02 + chain-09 (4 cases, both "
    "multi_hop, auth + inventory) picks the best checkpoint; the remaining 8 chains "
    "(9 cases) train. No chunk or document crosses a split boundary."
)


@dataclass(frozen=True, slots=True)
class Pair:
    query: str
    passage: str
    label: int
    case_id: str
    chain_id: str
    source_path: str
    chunk_index: int
    pair_type: str


def _split_for(chain_id: str) -> str:
    if chain_id in TEST_CHAINS:
        return "test"
    if chain_id in DEV_CHAINS:
        return "dev"
    return "train"


def _chunk_documents(
    documents: list[DocumentRecord], *, chunk_size: int, chunk_overlap: int
) -> dict[str, list[Chunk]]:
    return {
        doc.source_path: chunk_text(doc.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for doc in documents
    }


def _positives(case: GroundTruthCase, chunk_map: dict[str, list[Chunk]]) -> tuple[list[Pair], int]:
    pairs: list[Pair] = []
    n_fallback = 0
    snippets = [s.lower() for s in case.required_evidence_snippets]
    for source_path in case.required_document_source_paths:
        chunks = chunk_map.get(source_path)
        if not chunks:
            continue
        hits = [c for c in chunks if any(s in c.content.lower() for s in snippets)]
        chosen = hits or chunks[:1]
        if not hits:
            n_fallback += 1
        for chunk in chosen:
            pairs.append(
                Pair(
                    query=case.query,
                    passage=chunk.content,
                    label=1,
                    case_id=case.case_id,
                    chain_id=case.chain_id or "",
                    source_path=source_path,
                    chunk_index=chunk.index,
                    pair_type="positive",
                )
            )
    return pairs, n_fallback


def _negatives(
    case: GroundTruthCase,
    chunk_map: dict[str, list[Chunk]],
    documents_by_service: dict[str, list[str]],
    all_source_paths: list[str],
    rng: random.Random,
) -> list[Pair]:
    required = set(case.required_document_source_paths)
    pairs: list[Pair] = []

    same_service = [
        p for p in documents_by_service.get(case.service_name or "", []) if p not in required
    ]
    hard_pool = list(same_service)
    rng.shuffle(hard_pool)
    for source_path in hard_pool[:HARD_NEGATIVES_PER_CASE]:
        chunks = chunk_map.get(source_path) or []
        if not chunks:
            continue
        chunk = rng.choice(chunks)
        pairs.append(
            Pair(
                query=case.query,
                passage=chunk.content,
                label=0,
                case_id=case.case_id,
                chain_id=case.chain_id or "",
                source_path=source_path,
                chunk_index=chunk.index,
                pair_type="hard_negative_same_service",
            )
        )

    other_service_pool = [
        p for p in all_source_paths if p not in required and p not in same_service
    ]
    rng.shuffle(other_service_pool)
    for source_path in other_service_pool[:EASY_NEGATIVES_PER_CASE]:
        chunks = chunk_map.get(source_path) or []
        if not chunks:
            continue
        chunk = rng.choice(chunks)
        pairs.append(
            Pair(
                query=case.query,
                passage=chunk.content,
                label=0,
                case_id=case.case_id,
                chain_id=case.chain_id or "",
                source_path=source_path,
                chunk_index=chunk.index,
                pair_type="easy_negative_other_service",
            )
        )
    return pairs


def build_pairs(corpus: Corpus, cases: list[GroundTruthCase]) -> tuple[dict[str, list[Pair]], int]:
    settings = get_settings()
    chunk_map = _chunk_documents(
        corpus.documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    documents_by_service: dict[str, list[str]] = {}
    for doc in corpus.documents:
        if doc.service_name:
            documents_by_service.setdefault(doc.service_name, []).append(doc.source_path)
    all_source_paths = [doc.source_path for doc in corpus.documents]

    by_split: dict[str, list[Pair]] = {"train": [], "dev": [], "test": []}
    n_fallback_total = 0
    for case in cases:
        if not case.answerable or case.chain_id is None:
            continue
        split = _split_for(case.chain_id)
        positives, n_fallback = _positives(case, chunk_map)
        n_fallback_total += n_fallback
        rng = random.Random(f"{SEED}:{case.case_id}")
        negatives = _negatives(case, chunk_map, documents_by_service, all_source_paths, rng)
        by_split[split].extend(positives)
        by_split[split].extend(negatives)
    return by_split, n_fallback_total


def _write_jsonl(path: Path, pairs: list[Pair]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(
                json.dumps(
                    {
                        "query": pair.query,
                        "passage": pair.passage,
                        "label": pair.label,
                        "case_id": pair.case_id,
                        "chain_id": pair.chain_id,
                        "source_path": pair.source_path,
                        "chunk_index": pair.chunk_index,
                        "pair_type": pair.pair_type,
                    }
                )
                + "\n"
            )


def main() -> int:
    corpus = load_corpus(DATA_DIR)
    cases = load_ground_truth(DATA_DIR)
    by_split, n_fallback = build_pairs(corpus, cases)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, pairs in by_split.items():
        _write_jsonl(OUT_DIR / f"{split}.jsonl", pairs)

    (OUT_DIR / "split.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "test_chains": sorted(TEST_CHAINS),
                "dev_chains": sorted(DEV_CHAINS),
                "hard_negatives_per_case": HARD_NEGATIVES_PER_CASE,
                "easy_negatives_per_case": EASY_NEGATIVES_PER_CASE,
                "rationale": SPLIT_RATIONALE,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for split, pairs in by_split.items():
        n_pos = sum(1 for p in pairs if p.label == 1)
        n_neg = len(pairs) - n_pos
        n_cases = len({p.case_id for p in pairs})
        print(f"{split:>5}: {len(pairs):3d} pairs  ({n_pos} pos / {n_neg} neg, {n_cases} cases)")
    print(f"fallback positives (no snippet matched a chunk, used chunk 0): {n_fallback}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
