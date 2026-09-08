"""Reciprocal Rank Fusion — combine several ranked lists into one.

RRF (Cormack, Clarke & Buettcher, 2009) scores each item as
``sum(1 / (k + rank_i))`` over every list *i* it appears in, where ``rank_i``
is its 1-indexed position in that list and ``k`` is a smoothing constant
(default 60, the value from the paper). Items are then ordered by descending
score.

It uses only rank *position*, never the per-list scores, so it fuses a
bounded cosine-similarity list and an unbounded ``ts_rank_cd`` list without any
score normalisation — which is exactly why it is the default fusion for hybrid
retrieval here. A hand-verifiable pure function; see
``tests/unit/test_fusion.py``.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion[T: Hashable](
    ranked_lists: Sequence[Sequence[T]], *, k: int = DEFAULT_RRF_K
) -> list[tuple[T, float]]:
    """Fuse ``ranked_lists`` (each best-first) into one best-first list.

    Returns ``(item, rrf_score)`` pairs sorted by descending score. Ties are
    broken by first appearance across the input lists, so the result is fully
    deterministic regardless of whether ``T`` is orderable.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    scores: dict[T, float] = {}
    first_seen: dict[T, int] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            if item not in scores:
                scores[item] = 0.0
                first_seen[item] = len(first_seen)
            scores[item] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
