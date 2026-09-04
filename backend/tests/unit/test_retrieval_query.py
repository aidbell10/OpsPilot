from __future__ import annotations

import pytest

from opspilot.retrieval.semantic import _distance_to_score

pytestmark = pytest.mark.unit


def test_distance_to_score_bounds() -> None:
    assert _distance_to_score(0.0) == 1.0
    assert _distance_to_score(2.0) == 0.0
    assert _distance_to_score(1.0) == pytest.approx(0.5)


def test_distance_to_score_clamps_negative_distance() -> None:
    # cosine distance should never be negative, but the score must stay in [0, 1]
    assert _distance_to_score(-0.1) <= 1.0
