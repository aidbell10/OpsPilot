"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from opspilot.db.session import get_db
from opspilot.schemas.health import HealthResponse, ReadinessCheck, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: the process is up and serving."""
    return HealthResponse()


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(response: Response, db: Session = Depends(get_db)) -> ReadinessResponse:
    """Readiness: dependencies required to serve traffic are available."""
    checks: list[ReadinessCheck] = []

    # 1. database connectivity
    try:
        db.execute(text("SELECT 1"))
        checks.append(ReadinessCheck(name="database", ok=True))
    except Exception as exc:
        checks.append(ReadinessCheck(name="database", ok=False, detail=str(exc)[:200]))

    # 2. pgvector extension present
    try:
        row = db.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).first()
        if row is not None:
            checks.append(ReadinessCheck(name="pgvector", ok=True, detail=f"v{row[0]}"))
        else:
            checks.append(
                ReadinessCheck(name="pgvector", ok=False, detail="extension not installed")
            )
    except Exception as exc:
        checks.append(ReadinessCheck(name="pgvector", ok=False, detail=str(exc)[:200]))

    # 3. schema migrated (alembic_version table exists and is populated)
    try:
        row = db.execute(text("SELECT version_num FROM alembic_version")).first()
        if row is not None:
            checks.append(ReadinessCheck(name="migrations", ok=True, detail=row[0]))
        else:
            checks.append(ReadinessCheck(name="migrations", ok=False, detail="no version row"))
    except Exception:
        checks.append(ReadinessCheck(name="migrations", ok=False, detail="alembic_version missing"))

    result = ReadinessResponse.from_checks(checks)
    if result.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
