"""Incident investigation endpoints (Phase 5: strategy-configurable hybrid RAG)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from opspilot.config import get_settings
from opspilot.db.session import get_db
from opspilot.generation.parser import run_generation
from opspilot.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from opspilot.models.enums import Environment, RetrievalStrategy
from opspilot.models.incident import Incident
from opspilot.models.service import Service
from opspilot.providers.factory import get_embedding_provider, get_llm_provider
from opspilot.retrieval.filters import ChunkFilters
from opspilot.retrieval.semantic import search_historical_incidents
from opspilot.retrieval.strategy import retrieve_chunks
from opspilot.schemas.analysis import AnalyzeResponse, EvidenceItem, RelatedIncident

router = APIRouter(prefix="/incidents", tags=["incidents"])


class AnalyzeRequest(BaseModel):
    description: str = Field(min_length=10, max_length=8000)
    service: str | None = None
    version: str | None = None
    environment: Environment | None = None


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_incident(request: AnalyzeRequest, db: Session = Depends(get_db)) -> AnalyzeResponse:
    settings = get_settings()
    embedding_provider = get_embedding_provider()
    llm_provider = get_llm_provider()
    strategy = RetrievalStrategy.from_name(settings.retrieval_strategy)

    query_vector = embedding_provider.embed([request.description]).vectors[0]
    chunk_matches = retrieve_chunks(
        db,
        strategy=strategy,
        query_embedding=query_vector,
        query_text=request.description,
        top_k=settings.retrieval_top_k,
        filters=ChunkFilters(service_name=request.service),
        candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.rrf_k,
    )
    historical_matches = search_historical_incidents(
        db,
        query_vector,
        top_k=min(5, settings.retrieval_top_k),
        service_name=request.service,
    )

    evidence = [
        EvidenceItem(
            chunk_id=str(m.chunk_id),
            document_id=str(m.document_id),
            title=m.title,
            source_path=m.source_path,
            document_type=m.document_type,
            service_name=m.service_name,
            version=m.version,
            content=m.content,
            score=m.score,
        )
        for m in chunk_matches
    ]
    related_incidents = [
        RelatedIncident(
            incident_id=str(m.incident_id),
            title=m.title,
            service_name=m.service_name,
            occurred_at=m.occurred_at,
            severity=m.severity,
            root_cause=m.root_cause,
            resolution=m.resolution,
            score=m.score,
        )
        for m in historical_matches
    ]

    user_prompt = build_user_prompt(
        description=request.description,
        service=request.service,
        version=request.version,
        environment=request.environment.value if request.environment else None,
        evidence=chunk_matches,
        related_incidents=related_incidents,
    )
    analysis = run_generation(
        llm_provider,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        valid_chunk_ids={e.chunk_id for e in evidence},
        max_tokens=settings.llm_max_tokens,
    )

    service_id = None
    if request.service:
        service_id = db.execute(
            select(Service.id).where(Service.name == request.service)
        ).scalar_one_or_none()

    title = (
        request.description
        if len(request.description) <= 297
        else request.description[:297] + "..."
    )
    incident = Incident(
        id=uuid.uuid4(),
        title=title,
        description=request.description,
        service_id=service_id,
        reported_service=request.service,
        reported_version=request.version,
        environment=request.environment,
        analysis=analysis.model_dump(),
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)

    return AnalyzeResponse(
        incident_id=str(incident.id),
        analysis=analysis,
        evidence=evidence,
        related_incidents=related_incidents,
        retrieval_top_k=settings.retrieval_top_k,
        retrieval_strategy=strategy.value,
    )
