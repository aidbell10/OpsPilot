"""Typed records the generator emits.

These deliberately mirror the ORM in ``backend/src/opspilot/models`` (Phase 1)
without importing it — the generator stays dependency-free and the two can be
diffed in review. Phase 3 ingestion maps these onto real rows and assigns UUIDs;
stable string keys here (``source_path``, ``case_id``) are what the evaluation
harness references.

String value sets (``document_type``, ``environment``, ``severity``,
``difficulty``, deployment ``status``) match
``opspilot.models.enums`` exactly; ``tests/test_enums_match.py`` enforces it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

JsonDict = dict[str, Any]

DOCUMENT_TYPES = ("runbook", "postmortem", "architecture", "deployment", "known_error")
ENVIRONMENTS = ("production", "staging", "development")
SEVERITIES = ("sev1", "sev2", "sev3", "sev4")
DIFFICULTIES = ("straightforward", "multi_hop", "unanswerable", "adversarial")
DEPLOYMENT_STATUSES = ("succeeded", "failed", "rolled_back")


@dataclass(frozen=True, slots=True)
class ServiceRecord:
    name: str
    description: str
    repo_url: str
    tier: int
    owning_team: str
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeploymentRecord:
    service_name: str
    version: str
    environment: str
    deployed_at: str  # ISO-8601, UTC
    status: str
    change_summary: str
    changed_components: JsonDict  # {"functions": [...], "config": [...], "dependencies": [...]}
    is_planted_trigger: bool = False
    is_planted_fix: bool = False

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    doc_id: str
    document_type: str
    title: str
    source_path: str
    content: str
    service_name: str | None = None
    version: str | None = None
    environment: str | None = None
    doc_timestamp: str | None = None
    meta: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HistoricalIncidentRecord:
    incident_id: str
    title: str
    service_name: str
    occurred_at: str
    severity: str
    symptoms: str
    root_cause: str
    resolution: str
    related_deployment_version: str | None = None
    linked_document_source_path: str | None = None
    meta: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GroundTruthCase:
    """One evaluatable question with known-correct evidence.

    Feeds ``evaluation_cases`` in Phase 4. ``required_document_source_paths``
    and ``required_evidence_snippets`` are resolved to real chunk / document
    ids after ingestion.
    """

    case_id: str
    chain_id: str | None
    category: str
    difficulty: str
    answerable: bool
    query: str
    service_name: str | None
    expected_root_cause: str | None
    required_document_source_paths: list[str]
    required_evidence_snippets: list[str]
    acceptable_actions: list[str]
    forbidden_claims: list[str]
    related_deployment_version: str | None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Chain:
    """The full planted causal chain, kept for Phase 7 hard-negative mining."""

    chain_id: str
    service_name: str
    kind: str
    error_code: str
    changed_function: str
    dependency: str | None
    config_key: str | None
    trigger_deployment_version: str
    fix_deployment_version: str
    incident_id: str
    difficulty: str
    symptom_log_lines: list[str]
    document_source_paths: JsonDict
    narrative: JsonDict

    def to_dict(self) -> JsonDict:
        return asdict(self)
