"""The fictional company's services."""

from __future__ import annotations

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from opspilot.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Service(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "services"

    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    repo_url: Mapped[str | None] = mapped_column(String(300), default=None)
    tier: Mapped[int] = mapped_column(default=2)  # 1 = critical path
    owning_team: Mapped[str] = mapped_column(String(100), default="platform")
    # Names of other services this one calls directly. Present in the Phase 2
    # generator's ServiceRecord since Phase 1 but never persisted until the
    # Phase 8 agent's get_service_dependencies tool needed a real column
    # (migration 0002) instead of re-reading data/generated/services.json.
    depends_on: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
