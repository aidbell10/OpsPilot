from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow]

pytest.importorskip("sentence_transformers", reason="requires the 'ml' extra")


def test_cross_encoder_ranks_the_relevant_passage_first() -> None:
    from opspilot.providers.local_rerank import LocalCrossEncoderProvider

    provider = LocalCrossEncoderProvider()
    query = "why does checkout return HTTP 500 after a promotion code change"
    docs = [
        "The nightly inventory restock job reorders items below their threshold.",
        "validate_cart raises a 500 when a promotion code is present but empty after the deploy.",
        "Auth rotates signing keys every 24 hours across all regions.",
    ]
    scores = provider.rerank(query, docs).scores
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[1] == max(scores)


def test_cross_encoder_empty_documents() -> None:
    from opspilot.providers.local_rerank import LocalCrossEncoderProvider

    assert LocalCrossEncoderProvider().rerank("q", []).scores == []
