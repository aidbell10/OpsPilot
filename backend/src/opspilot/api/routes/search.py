"""Evidence search endpoint.

Placeholder until Phase 3. Returns 501; contract declared for the frontend.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/search", tags=["search"])

_NOT_IMPLEMENTED = "Search lands in Phase 3 (vector) / Phase 5 (hybrid)."


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
    service: str | None = None
    document_type: str | None = None


@router.post("", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def search(_request: SearchRequest) -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, _NOT_IMPLEMENTED)
