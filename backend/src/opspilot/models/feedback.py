"""End-user feedback on investigation results."""

from __future__ import annotations

import uuid

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from opspilot.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class UserFeedback(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "user_feedback"

    incident_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), default=None, index=True
    )
    case_id: Mapped[str | None] = mapped_column(String(64), default=None)

    rating: Mapped[int | None] = mapped_column(Integer, default=None)  # 1..5
    helpful: Mapped[bool | None] = mapped_column(Boolean, default=None)
    comment: Mapped[str] = mapped_column(Text, default="")
