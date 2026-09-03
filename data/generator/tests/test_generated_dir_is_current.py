"""If someone has a ``data/generated/`` checkout, make sure it is not stale.

The directory is git-ignored, so this is a no-op in CI and a safety net locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data.generator.emit import check
from data.generator.spec import DEFAULT_SEED, GeneratorSpec

_GENERATED = Path(__file__).resolve().parents[3] / "data" / "generated"


@pytest.mark.skipif(not _GENERATED.exists(), reason="no local data/generated/ checkout")
def test_local_corpus_matches_default_seed() -> None:
    problems = check(GeneratorSpec(seed=DEFAULT_SEED), _GENERATED)
    assert not problems, "run: python -m data.generator\n" + "\n".join(problems)
