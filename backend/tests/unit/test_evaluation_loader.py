from __future__ import annotations

import json
from pathlib import Path

import pytest

from opspilot.evaluation.loader import GroundTruthCase, dataset_version_from_manifest

pytestmark = pytest.mark.unit


def test_ground_truth_case_parses_full_real_shape() -> None:
    raw = {
        "acceptable_actions": ["Roll back to v5.0.5"],
        "answerable": True,
        "case_id": "gt-01-payments-auth-timeout-01",
        "category": "root_cause",
        "chain_id": "chain-01-payments-auth-timeout",
        "difficulty": "multi_hop",
        "expected_root_cause": "payments v5.0.6 changed authorize_payment ...",
        "forbidden_claims": ["the payments database was corrupted"],
        "query": "Payment authorizations started failing with PAY-50231. Why?",
        "related_deployment_version": "v5.0.6",
        "required_document_source_paths": ["postmortems/payments/2024-12-29-upstream-timeout.md"],
        "required_evidence_snippets": ["PAY-50231", "authorize_payment"],
        "service_name": "payments",
    }
    case = GroundTruthCase(**raw)
    assert case.case_id == "gt-01-payments-auth-timeout-01"
    assert case.answerable is True
    assert case.difficulty == "multi_hop"
    assert case.required_document_source_paths == [
        "postmortems/payments/2024-12-29-upstream-timeout.md"
    ]
    assert case.required_evidence_snippets == ["PAY-50231", "authorize_payment"]
    assert case.chain_id == "chain-01-payments-auth-timeout"


def test_ground_truth_case_parses_unanswerable_shape_with_null_fields() -> None:
    raw = {
        "acceptable_actions": ["State that the available evidence is insufficient to answer"],
        "answerable": False,
        "case_id": "unans-01-no-search-service",
        "category": "abstention",
        "chain_id": None,
        "difficulty": "unanswerable",
        "expected_root_cause": None,
        "forbidden_claims": ["the search service"],
        "query": "The search service is returning HTTP 500s. What's the root cause?",
        "related_deployment_version": None,
        "required_document_source_paths": [],
        "required_evidence_snippets": [],
        "service_name": None,
    }
    case = GroundTruthCase(**raw)
    assert case.answerable is False
    assert case.chain_id is None
    assert case.required_document_source_paths == []
    assert case.service_name is None


def test_dataset_version_from_manifest(tmp_path: Path) -> None:
    manifest = {"generator_version": "2.0.0", "seed": 42}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert dataset_version_from_manifest(tmp_path) == "gen2.0.0-seed42"


def test_dataset_version_falls_back_when_no_manifest(tmp_path: Path) -> None:
    assert dataset_version_from_manifest(tmp_path) == "unknown"
