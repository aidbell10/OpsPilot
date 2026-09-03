from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from opspilot.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture
def client(migrated_engine: Engine) -> TestClient:  # fixture ensures the DB is migrated
    from opspilot.config import get_settings
    from opspilot.db.session import reset_engine_cache

    get_settings.cache_clear()
    reset_engine_cache()
    return TestClient(create_app())


def test_readiness_is_green_against_migrated_db(client: TestClient) -> None:
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    names = {c["name"]: c["ok"] for c in body["checks"]}
    assert names == {"database": True, "pgvector": True, "migrations": True}
