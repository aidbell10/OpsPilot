"""The generator's string value sets must stay in lock-step with the backend
ORM enums. Skipped when the backend package is not importable (e.g. the
stdlib-only CI generator job)."""

from __future__ import annotations

import pytest

from data.generator import schema

enums = pytest.importorskip("opspilot.models.enums")


def _values(name: str) -> set[str]:
    return {member.value for member in getattr(enums, name)}


def test_document_types_match() -> None:
    assert set(schema.DOCUMENT_TYPES) == _values("DocumentType")


def test_environments_match() -> None:
    assert set(schema.ENVIRONMENTS) == _values("Environment")


def test_severities_match() -> None:
    assert set(schema.SEVERITIES) == _values("Severity")


def test_deployment_statuses_match() -> None:
    assert set(schema.DEPLOYMENT_STATUSES) == _values("DeploymentStatus")


def test_difficulties_match() -> None:
    assert set(schema.DIFFICULTIES) == _values("Difficulty")
