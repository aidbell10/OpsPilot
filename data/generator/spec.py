"""The knobs for one generation run.

A run is fully described by ``GeneratorSpec`` and nothing else (in particular:
no ``datetime.now``). The world's clock is ``world_start`` .. ``world_end``,
entirely synthetic.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

DEFAULT_SEED = 42

# The synthetic company's observable history. Deployments, incidents and
# document timestamps all fall inside this window.
WORLD_START = dt.date(2024, 9, 2)
WORLD_END = dt.date(2025, 8, 25)


@dataclass(frozen=True, slots=True)
class GeneratorSpec:
    seed: int = DEFAULT_SEED
    world_start: dt.date = WORLD_START
    world_end: dt.date = WORLD_END

    # Releases per service across the whole window (min, max). Each release is
    # a row in ``deployments``.
    releases_per_service: tuple[int, int] = (9, 14)

    # Historical incidents that are *not* attached to a planted ground-truth
    # chain — realistic noise so retrieval has distractors.
    background_incidents: tuple[int, int] = (11, 15)

    # Extra service-level runbooks beyond the per-chain runbooks
    # (deploy/rollback, on-call, scaling, ...).
    generic_runbooks_per_service: tuple[int, int] = (2, 3)

    # Deployment records notable enough to also get a prose "release notes"
    # document (type=deployment).
    deployment_docs: int = 12
