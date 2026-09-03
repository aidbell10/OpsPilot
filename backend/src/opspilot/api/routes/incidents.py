"""Incident investigation endpoints.

Placeholder until Phase 3 (baseline RAG). The contract is declared here so the
OpenAPI schema and frontend can be developed against it; calls return 501.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/incidents", tags=["incidents"])

_NOT_IMPLEMENTED = "Incident analysis lands in Phase 3 (baseline RAG)."


class AnalyzeRequest(BaseModel):
    description: str = Field(min_length=10, max_length=8000)
    service: str | None = None
    version: str | None = None
    environment: str | None = None


@router.post("/analyze", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def analyze_incident(_request: AnalyzeRequest) -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, _NOT_IMPLEMENTED)
