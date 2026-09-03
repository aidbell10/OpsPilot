"""Deployment / change records."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from opspilot.db.base import Base, JsonDict, TimestampMixin, UUIDPrimaryKeyMixin
from opspilot.db.types import str_enum
from opspilot.models.enums import DeploymentStatus, Environment


class Deployment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "deployments"

    service_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("services.id", ondelete="SET NULL"), default=None
    )
    service_name: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[str] = mapped_column(String(50), index=True)
    environment: Mapped[Environment] = mapped_column(
        str_enum(Environment), default=Environment.PRODUCTION
    )
    deployed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[DeploymentStatus] = mapped_column(
        str_enum(DeploymentStatus), default=DeploymentStatus.SUCCEEDED
    )
    change_summary: Mapped[str] = mapped_column(Text, default="")
    changed_components: Mapped[JsonDict] = mapped_column(default=dict)  # {"functions": [...], ...}

    __table_args__ = (
        Index("uq_deployments_service_name_version", "service_name", "version", unique=True),
    )
