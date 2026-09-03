from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from opspilot.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.unit
def test_health_is_live(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "opspilot-api"
    assert "x-request-id" in resp.headers


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/incidents/analyze", {"description": "checkout 500s after v2.14.0 deploy"}),
        ("post", "/search", {"query": "promotion validation error"}),
        ("post", "/feedback", {"comment": "great"}),
    ],
)
def test_phase3_endpoints_declared_but_not_implemented(
    client: TestClient, method: str, path: str, payload: dict[str, object]
) -> None:
    resp = getattr(client, method)(path, json=payload)
    assert resp.status_code == 501


@pytest.mark.unit
def test_openapi_lists_the_full_contract(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/health/ready", "/incidents/analyze", "/search", "/feedback"} <= set(paths)
