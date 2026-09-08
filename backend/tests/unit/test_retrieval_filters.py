from __future__ import annotations

import datetime as dt

import pytest

from opspilot.models.enums import DocumentType, Environment
from opspilot.retrieval.filters import ChunkFilters
from opspilot.retrieval.lexical import _rank_to_score

pytestmark = pytest.mark.unit


def test_empty_filter_produces_no_clauses() -> None:
    f = ChunkFilters()
    assert f.is_empty is True
    assert f.clauses() == []


def test_each_field_adds_one_clause() -> None:
    f = ChunkFilters(
        service_name="checkout",
        document_type=DocumentType.RUNBOOK,
        version="v2.14.0",
        environment=Environment.PRODUCTION,
        timestamp_after=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        timestamp_before=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
    )
    assert f.is_empty is False
    assert len(f.clauses()) == 6


def test_partial_filter() -> None:
    f = ChunkFilters(service_name="auth")
    assert f.is_empty is False
    assert len(f.clauses()) == 1


def test_rank_to_score_is_bounded() -> None:
    assert _rank_to_score(0.0) == 0.0
    assert _rank_to_score(1.0) == pytest.approx(0.5)
    assert 0.0 < _rank_to_score(1e6) < 1.0
