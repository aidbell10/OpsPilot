"""Resolved historical incidents / postmortems used for similarity lookup."""

from __future__ import annotations

import datetime as dt
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from opspilot.db.base import Base, JsonDict, TimestampMixin, UUIDPrimaryKeyMixin
from opspilot.db.types import str_enum
from opspilot.models.document import EMBEDDING_DIM
from opspilot.models.enums import Severity


class HistoricalIncident(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "historical_incidents"

    title: Mapped[str] = mapped_column(String(300))
    service_name: Mapped[str] = mapped_column(String(100), index=True)
    occurred_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    severity: Mapped[Severity] = mapped_column(str_enum(Severity), default=Severity.SEV3)

    symptoms: Mapped[str] = mapped_column(Text)
    root_cause: Mapped[str] = mapped_column(Text)
    resolution: Mapped[str] = mapped_column(Text)

    related_deployment_version: Mapped[str | None] = mapped_column(String(50), default=None)
    linked_document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), default=None
    )
    meta: Mapped[JsonDict] = mapped_column(default=dict)

    # Populated during ingestion (Phase 3). Nullable until then.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), default=None)
