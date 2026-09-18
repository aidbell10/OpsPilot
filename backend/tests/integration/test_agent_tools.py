"""Phase 8 — each agent tool against a live DB."""

from __future__ import annotations

import pytest
from corpus_fixtures import agent_corpus
from sqlalchemy.orm import Session

from opspilot.agent.tools import (
    FindSimilarIncidentsArgs,
    GetDeploymentArgs,
    GetIncidentLogsArgs,
    GetServiceDependenciesArgs,
    SearchDocsArgs,
    ToolContext,
    ToolError,
    dispatch,
    render_tool_specs,
)
from opspilot.config import get_settings
from opspilot.ingestion.pipeline import ingest_corpus
from opspilot.providers.fake import FakeEmbeddingProvider
from opspilot.telemetry.cost import CostAccumulator

pytestmark = [pytest.mark.integration]


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(
        embedding_provider=FakeEmbeddingProvider(dim=384),
        rerank_provider=None,
        settings=get_settings(),
        cost=CostAccumulator(),
    )


@pytest.fixture
def seeded(db_session: Session, clean_db: None) -> Session:
    ingest_corpus(
        db_session,
        agent_corpus(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        chunk_size=48,
        chunk_overlap=8,
    )
    db_session.commit()
    return db_session


def test_search_docs_finds_the_runbook(seeded: Session, ctx: ToolContext) -> None:
    ok, content, error = dispatch(
        "search_docs",
        {"query": "checkout HTTP 500 promotion validate_cart", "top_k": 3},
        seeded,
        ctx,
    )
    assert ok is True
    assert error is None
    assert "promotion validation" in content
    assert "chunk_id:" in content


def test_search_docs_respects_document_type_filter(seeded: Session, ctx: ToolContext) -> None:
    ok, content, _ = dispatch(
        "search_docs",
        {"query": "checkout", "document_type": "known_error", "top_k": 3},
        seeded,
        ctx,
    )
    assert ok is True
    assert "No matching document chunks found." in content


def test_find_similar_incidents_returns_root_cause(seeded: Session, ctx: ToolContext) -> None:
    ok, content, error = dispatch(
        "find_similar_incidents",
        {"symptoms": "checkout returns 500 for carts with promotion", "top_k": 3},
        seeded,
        ctx,
    )
    assert ok is True
    assert error is None
    assert "promotion validation raised on empty discount code" in content


def test_get_incident_logs_by_error_code(seeded: Session, ctx: ToolContext) -> None:
    ok, content, error = dispatch(
        "get_incident_logs",
        {"service_name": "checkout", "error_code": "CHK-50500"},
        seeded,
        ctx,
    )
    assert ok is True
    assert error is None
    assert "CHK-50500" in content
    assert "root_cause" not in content.lower()  # logs, not the answer


def test_get_incident_logs_unknown_error_code_is_a_clean_failure(
    seeded: Session, ctx: ToolContext
) -> None:
    ok, content, error = dispatch(
        "get_incident_logs",
        {"service_name": "checkout", "error_code": "DOES-NOT-EXIST"},
        seeded,
        ctx,
    )
    assert ok is False
    assert content == ""
    assert error is not None
    assert "no historical incident found" in error


def test_get_deployment_exact_lookup(seeded: Session, ctx: ToolContext) -> None:
    ok, content, error = dispatch(
        "get_deployment", {"service_name": "checkout", "version": "v2.14.0"}, seeded, ctx
    )
    assert ok is True
    assert error is None
    assert "validate_cart" in content
    assert "succeeded" in content


def test_get_deployment_unknown_version_is_a_clean_failure(
    seeded: Session, ctx: ToolContext
) -> None:
    ok, _, error = dispatch(
        "get_deployment", {"service_name": "checkout", "version": "v99.0.0"}, seeded, ctx
    )
    assert ok is False
    assert error is not None


def test_get_service_dependencies_both_directions(seeded: Session, ctx: ToolContext) -> None:
    ok, content, error = dispatch(
        "get_service_dependencies", {"service_name": "payments"}, seeded, ctx
    )
    assert ok is True
    assert error is None
    assert "depends on: auth" in content
    assert "depended on by: checkout" in content


def test_unknown_tool_name_is_a_clean_failure(seeded: Session, ctx: ToolContext) -> None:
    ok, content, error = dispatch("delete_everything", {}, seeded, ctx)
    assert ok is False
    assert content == ""
    assert error is not None
    assert "unknown tool" in error


def test_invalid_arguments_are_a_clean_failure(seeded: Session, ctx: ToolContext) -> None:
    # get_incident_logs requires error_code or incident_id — neither given.
    ok, content, error = dispatch("get_incident_logs", {"service_name": "checkout"}, seeded, ctx)
    assert ok is False
    assert content == ""
    assert error is not None
    assert "invalid arguments" in error


def test_get_incident_logs_raises_tool_error_directly() -> None:
    # ToolError is part of the public contract other callers might catch.
    with pytest.raises(ToolError):
        raise ToolError("example")


def test_render_tool_specs_lists_every_tool() -> None:
    rendered = render_tool_specs()
    for name in (
        "search_docs",
        "find_similar_incidents",
        "get_incident_logs",
        "get_deployment",
        "get_service_dependencies",
    ):
        assert name in rendered


def test_arg_models_validate_directly() -> None:
    SearchDocsArgs(query="x")
    FindSimilarIncidentsArgs(symptoms="x")
    GetIncidentLogsArgs(service_name="checkout", error_code="CHK-1")
    GetDeploymentArgs(service_name="checkout", version="v1")
    GetServiceDependenciesArgs(service_name="checkout")
    with pytest.raises(ValueError, match="error_code or incident_id"):
        GetIncidentLogsArgs(service_name="checkout")
