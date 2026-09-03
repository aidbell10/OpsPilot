"""Deterministic synthetic knowledge-base generator for OpsPilot (Phase 2).

Everything the retrieval / generation / evaluation stack is later measured
against is produced here from a single integer seed, with **no** wall-clock,
network, or filesystem reads. Same seed in, byte-identical corpus out.

Entry point::

    python -m data.generator --seed 42 --out data/generated

See ``data/generator/README.md`` for the design notes.
"""

from __future__ import annotations

# Bump when the *output* format or content changes in a way that should
# invalidate a previously generated corpus. Recorded in ``manifest.json``.
GENERATOR_VERSION = "2.0.0"

__all__ = ["GENERATOR_VERSION"]
