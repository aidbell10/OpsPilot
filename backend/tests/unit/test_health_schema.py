from __future__ import annotations

import pytest

from opspilot.schemas.health import HealthResponse, ReadinessCheck, ReadinessResponse


@pytest.mark.unit
def test_health_response_shape() -> None:
    payload = HealthResponse().model_dump()
    assert payload == {"status": "ok", "service": "opspilot-api", "version": payload["version"]}
    assert payload["version"]


@pytest.mark.unit
def test_readiness_ready_when_all_checks_pass() -> None:
    checks = [ReadinessCheck(name="database", ok=True), ReadinessCheck(name="pgvector", ok=True)]
    result = ReadinessResponse.from_checks(checks)
    assert result.status == "ready"


@pytest.mark.unit
def test_readiness_degraded_when_any_check_fails() -> None:
    checks = [
        ReadinessCheck(name="database", ok=True),
        ReadinessCheck(name="pgvector", ok=False, detail="missing"),
    ]
    result = ReadinessResponse.from_checks(checks)
    assert result.status == "degraded"
