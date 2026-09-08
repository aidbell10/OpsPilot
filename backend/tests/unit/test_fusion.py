from __future__ import annotations

import pytest

from opspilot.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion

pytestmark = pytest.mark.unit


def test_single_list_preserves_order_and_scores() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert [item for item, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] == pytest.approx(1 / 61)
    assert fused[1][1] == pytest.approx(1 / 62)
    assert fused[2][1] == pytest.approx(1 / 63)


def test_item_in_both_lists_outranks_a_higher_but_single_list_item() -> None:
    # "b" is rank 2 in both lists -> 1/62 + 1/62 ≈ 0.03226
    # "a" is rank 1 in one list only -> 1/61 ≈ 0.01639
    fused = reciprocal_rank_fusion([["a", "b"], ["x", "b"]], k=60)
    assert fused[0][0] == "b"
    assert fused[0][1] == pytest.approx(2 / 62)


def test_ties_break_by_first_appearance() -> None:
    # every item appears exactly once at rank 1 -> identical scores
    fused = reciprocal_rank_fusion([["a"], ["b"], ["c"]], k=60)
    assert [item for item, _ in fused] == ["a", "b", "c"]
    assert all(score == pytest.approx(1 / 61) for _, score in fused)


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        reciprocal_rank_fusion([["a"]], k=0)


def test_default_k_matches_the_paper() -> None:
    assert DEFAULT_RRF_K == 60
