"""User feedback endpoint (Phase 3)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from opspilot.db.session import get_db
from opspilot.models.feedback import UserFeedback
from opspilot.schemas.feedback import FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    incident_id: uuid.UUID | None = None
    case_id: str | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    helpful: bool | None = None
    comment: str = Field(default="", max_length=4000)


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)) -> FeedbackResponse:
    row = UserFeedback(
        id=uuid.uuid4(),
        incident_id=request.incident_id,
        case_id=request.case_id,
        rating=request.rating,
        helpful=request.helpful,
        comment=request.comment,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return FeedbackResponse(id=str(row.id), created_at=row.created_at)
