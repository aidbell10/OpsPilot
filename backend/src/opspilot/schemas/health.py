"""Health / readiness response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from opspilot import __version__


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["opspilot-api"] = "opspilot-api"
    version: str = __version__


class ReadinessCheck(BaseModel):
    name: str
    ok: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    checks: list[ReadinessCheck] = Field(default_factory=list)

    @classmethod
    def from_checks(cls, checks: list[ReadinessCheck]) -> ReadinessResponse:
        ok = all(c.ok for c in checks)
        return cls(status="ready" if ok else "degraded", checks=checks)
