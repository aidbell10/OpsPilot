from __future__ import annotations

import math

import pytest

from opspilot.evaluation.metrics import (
    AbstentionCounts,
    abstention_counts,
    citation_precision,
    evidence_coverage,
    hallucinated_forbidden_claim,
    mrr,
    ndcg_at_10,
    recall_at_k,
    sum_abstention_counts,
)

pytestmark = pytest.mark.unit


# --- recall_at_k -------------------------------------------------------------


def test_recall_at_k_vacuous_when_nothing_required() -> None:
    assert recall_at_k(["x", "y"], []) == 1.0


def test_recall_at_k_no_overlap() -> None:
    assert recall_at_k(["x", "y"], ["a", "b"]) == 0.0


def test_recall_at_k_partial_overlap() -> None:
    assert recall_at_k(["a", "y"], ["a", "b"]) == 0.5


def test_recall_at_k_full_overlap() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "b"]) == 1.0


# --- evidence_coverage --------------------------------------------------------


def test_evidence_coverage_vacuous_when_nothing_required() -> None:
    assert evidence_coverage(["some text"], []) == 1.0


def test_evidence_coverage_partial_case_insensitive() -> None:
    texts = ["Error PAY-50231 occurred during authorize_payment"]
    assert evidence_coverage(texts, ["pay-50231", "checkout"]) == 0.5


def test_evidence_coverage_full() -> None:
    texts = ["PAY-50231 in authorize_payment during v5.0.6"]
    assert evidence_coverage(texts, ["PAY-50231", "authorize_payment", "v5.0.6"]) == 1.0


# --- mrr -----------------------------------------------------------------------


def test_mrr_vacuous_when_nothing_required() -> None:
    assert mrr(["x", "y"], []) == 1.0


def test_mrr_first_rank() -> None:
    assert mrr(["a", "x"], ["a"]) == 1.0


def test_mrr_second_rank() -> None:
    assert mrr(["x", "a", "y", "b"], ["a", "b"]) == pytest.approx(0.5)


def test_mrr_not_found() -> None:
    assert mrr(["x", "y"], ["a"]) == 0.0


# --- ndcg_at_10 ------------------------------------------------------------


def test_ndcg_vacuous_when_nothing_required() -> None:
    assert ndcg_at_10(["x", "y"], []) == 1.0


def test_ndcg_hand_computed() -> None:
    # relevant docs "a" (rank 2) and "b" (rank 4); ranks are 1-indexed.
    retrieved = ["x", "a", "y", "b"]
    required = ["a", "b"]
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = dcg / idcg
    assert ndcg_at_10(retrieved, required) == pytest.approx(expected)


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_10(["a", "b", "x"], ["a", "b"]) == pytest.approx(1.0)


def test_ndcg_nothing_relevant_retrieved_is_zero() -> None:
    assert ndcg_at_10(["x", "y", "z"], ["a", "b"]) == 0.0


def test_ndcg_bounded_by_one_with_duplicate_document_chunks() -> None:
    # Multiple retrieved chunks can belong to the same required document
    # (e.g. 3 chunks from the one required doc "a"): the ideal ranking is
    # capped at the number of relevant retrieved *items*, not at
    # len(required) documents, so nDCG must never exceed 1.0.
    retrieved = ["a", "a", "a", "x", "y"]
    required = ["a"]
    assert 0.0 <= ndcg_at_10(retrieved, required) <= 1.0
    assert ndcg_at_10(retrieved, required) == pytest.approx(1.0)


# --- citation_precision -----------------------------------------------------


def test_citation_precision_undefined_with_no_citations() -> None:
    assert citation_precision([], ["a", "b"]) is None


def test_citation_precision_undefined_with_no_required_docs() -> None:
    assert citation_precision(["a"], []) is None


def test_citation_precision_partial() -> None:
    assert citation_precision(["a", "x"], ["a", "b"]) == pytest.approx(0.5)


def test_citation_precision_perfect() -> None:
    assert citation_precision(["a", "b"], ["a", "b", "c"]) == pytest.approx(1.0)


# --- hallucinated_forbidden_claim -------------------------------------------


def test_hallucination_detects_case_insensitive_substring() -> None:
    text = "We believe the Payments Database was corrupted during the outage."
    assert hallucinated_forbidden_claim(text, ["the payments database was corrupted"]) is True


def test_hallucination_false_when_absent() -> None:
    text = "auth v5.0.1 raised the JWKS cache TTL."
    assert hallucinated_forbidden_claim(text, ["the token signing key was compromised"]) is False


def test_hallucination_false_with_no_forbidden_claims() -> None:
    assert hallucinated_forbidden_claim("anything at all", []) is False


# --- abstention confusion matrix --------------------------------------------


def test_abstention_counts_true_positive() -> None:
    assert abstention_counts(should_abstain=True, did_abstain=True) == AbstentionCounts(
        true_positive=1
    )


def test_abstention_counts_false_positive() -> None:
    assert abstention_counts(should_abstain=False, did_abstain=True) == AbstentionCounts(
        false_positive=1
    )


def test_abstention_counts_false_negative() -> None:
    assert abstention_counts(should_abstain=True, did_abstain=False) == AbstentionCounts(
        false_negative=1
    )


def test_abstention_counts_true_negative() -> None:
    assert abstention_counts(should_abstain=False, did_abstain=False) == AbstentionCounts(
        true_negative=1
    )


def test_abstention_precision_recall_f1_hand_computed() -> None:
    # 1 unanswerable case correctly abstained (TP), 1 answerable case wrongly
    # abstained (FP): precision = 1/2, recall = 1/1, f1 = 2*0.5*1/(0.5+1).
    counts = sum_abstention_counts(
        [
            abstention_counts(should_abstain=True, did_abstain=True),
            abstention_counts(should_abstain=False, did_abstain=True),
        ]
    )
    assert counts.precision() == pytest.approx(0.5)
    assert counts.recall() == pytest.approx(1.0)
    assert counts.f1() == pytest.approx(2 / 3)


def test_abstention_precision_zero_when_never_abstains() -> None:
    counts = sum_abstention_counts([abstention_counts(should_abstain=True, did_abstain=False)])
    assert counts.precision() == 0.0
    assert counts.recall() == 0.0
    assert counts.f1() == 0.0
