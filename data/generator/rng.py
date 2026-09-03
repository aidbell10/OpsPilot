"""Namespaced deterministic randomness.

``random.Random`` seeded directly is fine, but a single shared stream makes the
generator fragile: inserting one extra ``.choice`` call early on reshuffles
every downstream document. Instead every part of the generator draws from its
own **named sub-stream**, derived from ``(seed, *names)`` via SHA-256. Adding a
new document type perturbs only its own stream.

Python's built-in ``hash()`` is salted per-process (``PYTHONHASHSEED``), so it is
never used for anything that must be reproducible.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def _seed_int(seed: int, parts: Sequence[object]) -> int:
    payload = "\x1f".join([str(seed), *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class Rng:
    """A deterministic random source that spawns independent named sub-streams."""

    __slots__ = ("_random", "_seed")

    def __init__(self, seed: int, *parts: object) -> None:
        self._seed = seed
        self._random = random.Random(_seed_int(seed, parts))

    def child(self, *parts: object) -> Rng:
        """A fresh, independent stream identified by ``parts``."""
        child = Rng.__new__(Rng)
        child._seed = self._seed
        child._random = random.Random(_seed_int(self._seed, parts))
        return child

    # --- thin, explicit wrappers (keeps call sites greppable) ----------------

    def choice(self, items: Sequence[T]) -> T:
        return items[self._random.randrange(len(items))]

    def sample(self, items: Sequence[T], k: int) -> list[T]:
        return self._random.sample(list(items), k)

    def shuffled(self, items: Sequence[T]) -> list[T]:
        out = list(items)
        self._random.shuffle(out)
        return out

    def int(self, low: int, high: int) -> int:
        """Inclusive on both ends."""
        return self._random.randint(low, high)

    def chance(self, p: float) -> bool:
        return self._random.random() < p

    def weighted(self, choices: Sequence[tuple[T, float]]) -> T:
        population = [c for c, _ in choices]
        weights = [w for _, w in choices]
        return self._random.choices(population, weights=weights, k=1)[0]
