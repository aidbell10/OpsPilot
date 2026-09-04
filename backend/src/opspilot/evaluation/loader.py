"""Load ground-truth evaluation cases (Phase 2 generator output) into ``evaluation_cases``.

Field reconciliation — the ``EvaluationCase`` ORM model (and ``docs/evaluation.md``) were
written in Phase 1, before the generator's exact ``ground_truth.json`` shape existed. Three
mismatches, resolved here rather than by migrating the schema:

* ``required_document_source_paths`` -> ``required_document_ids``. These are NOT UUIDs —
  they're ``Document.source_path`` natural keys, because ``Document`` ids don't exist (and
  aren't stable across re-ingests) until ingestion runs. Retrieval metrics
  (:mod:`opspilot.evaluation.metrics`) compare directly against
  ``ChunkMatch.source_path``, so no id-resolution lookup is needed at evaluation time either
  — the natural key round-trips end to end.
* ``required_evidence_snippets`` -> ``required_evidence_ids``. Repurposed: this column holds
  literal substrings expected to appear in relevant evidence text (used for evidence-coverage
  / grounding checks), not ids at all. The column name is a Phase-1 relic; renaming it would
  need a migration with no functional benefit, so the repurposing is documented here instead.
* ``chain_id`` (present in the source file) is not persisted — no metric groups by chain, and
  it is recoverable from ``case_id``'s prefix (e.g. ``gt-01-...``) if ever needed. Add a
  ``meta`` JSONB column in a later migration if that changes.

``split``: the generator emits no split field (22 cases total, in 12 chains — several cases
per chain, e.g. ``gt-01-payments-auth-timeout-{01,02,adv}``). All cases load as
``EvalSplit.DEV`` until the corpus grows toward the ~90/50 target in ``docs/evaluation.md``:
splitting today would either be statistically meaningless at this size, or risk leaking chain
siblings (near-duplicate queries about the same underlying incident) across dev/test, which
would invalidate a held-out test guarantee.

``dataset_version``: derived from ``data/generated/manifest.json`` as
``f"gen{generator_version}-seed{seed}"`` (e.g. ``gen2.0.0-seed42``) — stable across identical
rebuilds of the same corpus, and changes whenever the generator's output semantics or seed
change, matching the "versioned dataset" requirement in the roadmap.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from opspilot.models.enums import Difficulty, EvalSplit
from opspilot.models.evaluation import EvaluationCase


class GroundTruthCase(BaseModel):
    case_id: str
    query: str
    answerable: bool
    difficulty: str
    category: str = "general"
    service_name: str | None = None
    expected_root_cause: str | None = None
    acceptable_actions: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    required_document_source_paths: list[str] = Field(default_factory=list)
    required_evidence_snippets: list[str] = Field(default_factory=list)
    chain_id: str | None = None
    related_deployment_version: str | None = None


def load_ground_truth(data_dir: Path) -> list[GroundTruthCase]:
    raw: list[dict[str, Any]] = json.loads(
        (data_dir / "ground_truth.json").read_text(encoding="utf-8")
    )
    return [GroundTruthCase(**r) for r in raw]


def dataset_version_from_manifest(data_dir: Path) -> str:
    """Derive a stable dataset version from the generator's manifest.

    Falls back to ``"unknown"`` when no manifest is present (e.g. a hand-built
    fixture corpus in a test) rather than raising — dataset versioning is a
    provenance nicety, not a hard requirement for loading cases.
    """
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        return "unknown"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("generator_version", "0")
    seed = manifest.get("seed", "0")
    return f"gen{version}-seed{seed}"


def upsert_ground_truth(
    session: Session,
    cases: list[GroundTruthCase],
    *,
    dataset_version: str,
    split: EvalSplit = EvalSplit.DEV,
) -> int:
    """Idempotent upsert by the schema's ``(dataset_version, case_id)`` unique key."""
    for case in cases:
        existing = session.execute(
            select(EvaluationCase).where(
                EvaluationCase.dataset_version == dataset_version,
                EvaluationCase.case_id == case.case_id,
            )
        ).scalar_one_or_none()
        if existing is None:
            existing = EvaluationCase(
                id=uuid.uuid4(), dataset_version=dataset_version, case_id=case.case_id
            )
            session.add(existing)

        existing.split = split
        existing.query = case.query
        existing.answerable = case.answerable
        existing.service_name = case.service_name
        existing.required_document_ids = case.required_document_source_paths
        existing.required_evidence_ids = case.required_evidence_snippets
        existing.expected_root_cause = case.expected_root_cause
        existing.acceptable_actions = case.acceptable_actions
        existing.forbidden_claims = case.forbidden_claims
        existing.difficulty = Difficulty(case.difficulty)
        existing.category = case.category

    session.flush()
    return len(cases)
