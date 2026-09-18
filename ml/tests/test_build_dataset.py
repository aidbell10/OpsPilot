"""Unit tests for the pure dataset-building logic in ``ml/build_dataset.py``.

Hermetic: builds a tiny synthetic corpus in-memory rather than depending on
``data/generated`` (Phase 2) or a running Postgres, so these run as fast,
deterministic unit tests. Needs the backend's editable ``opspilot`` install
(``ml`` extra not required — only :mod:`opspilot.ingestion.chunking` and the
Pydantic record types are used) — run via
``cd backend && uv run pytest ../ml/tests -q`` (``make ml-test``).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_dataset import (
    DEV_CHAINS,
    TEST_CHAINS,
    _negatives,
    _positives,
    _split_for,
    build_pairs,
)
from opspilot.evaluation.loader import GroundTruthCase
from opspilot.ingestion.chunking import Chunk, chunk_text
from opspilot.ingestion.loader import Corpus, DocumentRecord


def _doc(source_path: str, content: str, service_name: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=source_path,
        document_type="runbook",
        title=source_path,
        content=content,
        service_name=service_name,
        source_path=source_path,
    )


def _case(
    case_id: str,
    *,
    chain_id: str | None,
    service_name: str,
    required: list[str],
    snippets: list[str],
    answerable: bool = True,
) -> GroundTruthCase:
    return GroundTruthCase(
        case_id=case_id,
        query=f"query for {case_id}",
        answerable=answerable,
        difficulty="straightforward",
        service_name=service_name,
        required_document_source_paths=required,
        required_evidence_snippets=snippets,
        chain_id=chain_id,
    )


def _chunk_map(*docs: DocumentRecord) -> dict[str, list[Chunk]]:
    return {d.source_path: chunk_text(d.content, chunk_size=512, chunk_overlap=64) for d in docs}


def test_split_for_known_chains() -> None:
    assert _split_for(next(iter(TEST_CHAINS))) == "test"
    assert _split_for(next(iter(DEV_CHAINS))) == "dev"
    assert _split_for("chain-not-in-either-set") == "train"


def test_positives_use_the_chunk_containing_the_snippet() -> None:
    doc = _doc("runbooks/pay.md", "Intro text.\n\nSee error ERR-999 for details.", "payments")
    case = _case(
        "c1",
        chain_id="chain-x",
        service_name="payments",
        required=[doc.source_path],
        snippets=["ERR-999"],
    )
    pairs, n_fallback = _positives(case, _chunk_map(doc))
    assert n_fallback == 0
    assert len(pairs) == 1
    assert pairs[0].label == 1
    assert "ERR-999" in pairs[0].passage
    assert pairs[0].source_path == doc.source_path


def test_positives_fall_back_to_first_chunk_when_snippet_absent() -> None:
    doc = _doc("runbooks/pay.md", "No matching token here at all.", "payments")
    case = _case(
        "c1",
        chain_id="chain-x",
        service_name="payments",
        required=[doc.source_path],
        snippets=["NOWHERE"],
    )
    pairs, n_fallback = _positives(case, _chunk_map(doc))
    assert n_fallback == 1
    assert len(pairs) == 1
    assert pairs[0].chunk_index == 0


def test_negatives_exclude_required_docs_and_respect_caps() -> None:
    required_doc = _doc("runbooks/req.md", "required content", "payments")
    same_service_docs = [
        _doc(f"runbooks/pay-{i}.md", f"payments distractor {i}", "payments") for i in range(8)
    ]
    other_service_docs = [
        _doc(f"runbooks/auth-{i}.md", f"auth distractor {i}", "auth") for i in range(4)
    ]
    all_docs = [required_doc, *same_service_docs, *other_service_docs]

    documents_by_service: dict[str, list[str]] = {}
    for d in all_docs:
        if d.service_name:
            documents_by_service.setdefault(d.service_name, []).append(d.source_path)

    case = _case(
        "c1",
        chain_id="chain-x",
        service_name="payments",
        required=[required_doc.source_path],
        snippets=[],
    )
    chunk_map = _chunk_map(*all_docs)
    rng = random.Random("test-seed")
    negatives = _negatives(
        case, chunk_map, documents_by_service, [d.source_path for d in all_docs], rng
    )

    hard = [n for n in negatives if n.pair_type == "hard_negative_same_service"]
    easy = [n for n in negatives if n.pair_type == "easy_negative_other_service"]
    assert len(hard) == 5  # HARD_NEGATIVES_PER_CASE
    assert len(easy) == 2  # EASY_NEGATIVES_PER_CASE
    assert all(n.label == 0 for n in negatives)
    assert all(n.source_path != required_doc.source_path for n in negatives)
    assert all(n.source_path.startswith("runbooks/pay-") for n in hard)
    assert all(n.source_path.startswith("runbooks/auth-") for n in easy)


def test_negatives_are_deterministic_for_a_given_seed() -> None:
    docs = [_doc(f"runbooks/pay-{i}.md", f"content {i}", "payments") for i in range(6)]
    documents_by_service = {"payments": [d.source_path for d in docs]}
    chunk_map = _chunk_map(*docs)
    case = _case("c1", chain_id="chain-x", service_name="payments", required=[], snippets=[])

    all_paths = [d.source_path for d in docs]
    first = _negatives(case, chunk_map, documents_by_service, all_paths, random.Random("s"))
    second = _negatives(case, chunk_map, documents_by_service, all_paths, random.Random("s"))
    assert [n.source_path for n in first] == [n.source_path for n in second]


def test_build_pairs_splits_by_chain_and_skips_unanswerable() -> None:
    test_chain = next(iter(TEST_CHAINS))
    dev_chain = next(iter(DEV_CHAINS))
    train_chain = "chain-not-in-either-set"

    docs = [
        _doc("runbooks/payments-req.md", "PAY-1 details here", "payments"),
        _doc("runbooks/payments-other.md", "unrelated payments doc", "payments"),
    ]
    corpus = Corpus(services=[], documents=docs, deployments=[], historical_incidents=[])
    cases = [
        _case(
            "train-1",
            chain_id=train_chain,
            service_name="payments",
            required=["runbooks/payments-req.md"],
            snippets=["PAY-1"],
        ),
        _case(
            "dev-1",
            chain_id=dev_chain,
            service_name="payments",
            required=["runbooks/payments-req.md"],
            snippets=["PAY-1"],
        ),
        _case(
            "test-1",
            chain_id=test_chain,
            service_name="payments",
            required=["runbooks/payments-req.md"],
            snippets=["PAY-1"],
        ),
        _case(
            "unanswerable-1",
            chain_id=None,
            service_name="payments",
            required=[],
            snippets=[],
            answerable=False,
        ),
    ]

    by_split, n_fallback = build_pairs(corpus, cases)
    assert n_fallback == 0
    assert {p.case_id for p in by_split["train"]} == {"train-1"}
    assert {p.case_id for p in by_split["dev"]} == {"dev-1"}
    assert {p.case_id for p in by_split["test"]} == {"test-1"}
    all_case_ids = {p.case_id for pairs in by_split.values() for p in pairs}
    assert "unanswerable-1" not in all_case_ids


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
