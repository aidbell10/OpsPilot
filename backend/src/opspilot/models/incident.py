"""Engineer-submitted incidents under investigation."""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from opspilot.db.base import Base, JsonDict, TimestampMixin, UUIDPrimaryKeyMixin
from opspilot.db.types import str_enum
from opspilot.models.enums import Environment, IncidentStatus


class Incident(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "incidents"

    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text)

    service_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("services.id", ondelete="SET NULL"), default=None
    )
    reported_service: Mapped[str | None] = mapped_column(String(100), default=None)
    reported_version: Mapped[str | None] = mapped_column(String(50), default=None)
    environment: Mapped[Environment | None] = mapped_column(str_enum(Environment), default=None)

    status: Mapped[IncidentStatus] = mapped_column(
        str_enum(IncidentStatus), default=IncidentStatus.OPEN
    )

    # Snapshot of the most recent analysis result (structured LLM output +
    # retrieval trace). Full history lives in evaluation_results / traces later.
    analysis: Mapped[JsonDict | None] = mapped_column(default=None)
