"""Structured LLM output contract, plus the API request/response shapes built on it.

``AnalysisResult`` is the schema the LLM is asked to fill in (see
``opspilot.generation.prompt``) and the schema its raw text is validated
against (see ``opspilot.generation.parser``) — the same model drives both the
prompt and the validation, so they can never drift apart.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

from opspilot.models.enums import DocumentType


class AnalysisResult(BaseModel):
    """The LLM's structured investigation result. Never trusted uncited."""

    hypothesis: str = Field(
        default="", description="A one- or two-sentence likely root-cause hypothesis."
    )
    root_cause: str | None = Field(
        default=None, description="A more detailed root-cause explanation, or null if unknown."
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the hypothesis, 0.0-1.0."
    )
    evidence_sufficient: bool = Field(
        default=False,
        description="True only if the cited evidence actually supports the hypothesis.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Chunk ids (exactly as shown in the evidence) that support the hypothesis.",
    )
    abstain_reason: str | None = Field(
        default=None,
        description="Required when evidence_sufficient is false: why the evidence falls short.",
    )


class EvidenceItem(BaseModel):
    """A retrieved chunk, shaped for display and for citation matching."""

    chunk_id: str
    document_id: str
    title: str
    source_path: str | None
    document_type: DocumentType
    service_name: str | None
    version: str | None
    content: str
    score: float


class RelatedIncident(BaseModel):
    incident_id: str
    title: str
    service_name: str
    occurred_at: dt.datetime
    severity: str
    root_cause: str
    resolution: str
    score: float


class AnalyzeResponse(BaseModel):
    incident_id: str
    analysis: AnalysisResult
    evidence: list[EvidenceItem]
    related_incidents: list[RelatedIncident]
    retrieval_top_k: int
