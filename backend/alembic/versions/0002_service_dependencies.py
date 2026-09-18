"""service dependencies

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-25

Adds ``services.depends_on`` (names of services this one calls directly).
Present in the Phase 2 generator's ``ServiceRecord`` since Phase 1 but never
persisted — the Phase 8 agent's ``get_service_dependencies`` tool needs a real
queryable column rather than re-reading ``data/generated/services.json``.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "services",
        sa.Column(
            "depends_on",
            postgresql.ARRAY(sa.String()),
            server_default="{}",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("services", "depends_on")
