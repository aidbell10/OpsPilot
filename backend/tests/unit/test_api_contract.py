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
def test_feedback_does_not_require_a_database_for_bad_input(client: TestClient) -> None:
    # comment alone with no incident/case id is a validation-shaped request;
    # a too-long comment must 422 before ever touching get_db.
    resp = client.post("/feedback", json={"comment": "x" * 5000})
    assert resp.status_code == 422


@pytest.mark.unit
def test_analyze_rejects_too_short_description(client: TestClient) -> None:
    resp = client.post("/incidents/analyze", json={"description": "short"})
    assert resp.status_code == 422


@pytest.mark.unit
def test_investigate_rejects_too_short_description(client: TestClient) -> None:
    resp = client.post("/incidents/investigate", json={"description": "short"})
    assert resp.status_code == 422


@pytest.mark.unit
def test_investigate_rejects_out_of_range_max_tool_calls(client: TestClient) -> None:
    resp = client.post(
        "/incidents/investigate",
        json={"description": "checkout is returning 500s", "max_tool_calls": 50},
    )
    assert resp.status_code == 422


@pytest.mark.unit
def test_search_rejects_bad_document_type(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "checkout 500s", "document_type": "not-a-type"})
    assert resp.status_code == 422


@pytest.mark.unit
def test_openapi_lists_the_full_contract(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/health",
        "/health/ready",
        "/incidents/analyze",
        "/incidents/investigate",
        "/search",
        "/feedback",
    } <= set(paths)
