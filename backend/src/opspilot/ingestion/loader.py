"""Load the synthetic knowledge base (Phase 2 generator output) from JSON.

Field names mirror ``data/generator/schema.py`` exactly; Pydantic validates
shape and types on load but does not otherwise transform the data — that
happens in :mod:`opspilot.ingestion.pipeline`.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ServiceRecord(BaseModel):
    name: str
    description: str = ""
    repo_url: str | None = None
    tier: int = 2
    owning_team: str = "platform"
    depends_on: list[str] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    doc_id: str
    document_type: str
    title: str
    content: str
    service_name: str | None = None
    version: str | None = None
    environment: str | None = None
    source_path: str
    doc_timestamp: dt.datetime | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class DeploymentRecord(BaseModel):
    service_name: str
    version: str
    environment: str
    deployed_at: dt.datetime
    status: str
    change_summary: str = ""
    changed_components: dict[str, Any] = Field(default_factory=dict)
    is_planted_trigger: bool = False
    is_planted_fix: bool = False


class HistoricalIncidentRecord(BaseModel):
    incident_id: str
    title: str
    service_name: str
    occurred_at: dt.datetime
    severity: str
    symptoms: str
    root_cause: str
    resolution: str
    related_deployment_version: str | None = None
    linked_document_source_path: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Corpus(BaseModel):
    services: list[ServiceRecord]
    documents: list[DocumentRecord]
    deployments: list[DeploymentRecord]
    historical_incidents: list[HistoricalIncidentRecord]


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def load_corpus(data_dir: Path) -> Corpus:
    """Load ``services/documents/deployments/historical_incidents`` from ``data_dir``.

    ``chains.json`` and ``ground_truth.json`` belong to the Phase 4 evaluation
    harness and are intentionally not loaded here.
    """
    return Corpus(
        services=[ServiceRecord(**r) for r in _read_json_list(data_dir / "services.json")],
        documents=[DocumentRecord(**r) for r in _read_json_list(data_dir / "documents.json")],
        deployments=[DeploymentRecord(**r) for r in _read_json_list(data_dir / "deployments.json")],
        historical_incidents=[
            HistoricalIncidentRecord(**r)
            for r in _read_json_list(data_dir / "historical_incidents.json")
        ],
    )
