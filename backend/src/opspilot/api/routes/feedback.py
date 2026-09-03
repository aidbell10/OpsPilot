"""User feedback endpoint.

Placeholder until Phase 3. Returns 501; contract declared for the frontend.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/feedback", tags=["feedback"])

_NOT_IMPLEMENTED = "Feedback persistence lands in Phase 3."


class FeedbackRequest(BaseModel):
    incident_id: str | None = None
    case_id: str | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    helpful: bool | None = None
    comment: str = Field(default="", max_length=4000)


@router.post("", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def submit_feedback(_request: FeedbackRequest) -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, _NOT_IMPLEMENTED)
