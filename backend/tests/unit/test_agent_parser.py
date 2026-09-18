from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from opspilot.agent.parser import parse_decision, verify_citations
from opspilot.schemas.agent import AgentDecision

pytestmark = pytest.mark.unit

KNOWN_OBS = {"obs-1", "obs-2"}


def _call_tool_payload(**overrides: object) -> str:
    payload: dict[str, object] = {
        "action": "call_tool",
        "thought": "need more evidence",
        "tool": "search_docs",
        "arguments": {"query": "PAY-50231"},
    }
    payload.update(overrides)
    return json.dumps(payload)


def _final_answer_payload(**overrides: object) -> str:
    payload: dict[str, object] = {
        "action": "final_answer",
        "thought": "enough evidence now",
        "hypothesis": "auth JWKS cache went stale",
        "root_cause": "AUTH_JWKS_CACHE_TTL_S increased beyond token lifetime",
        "confidence": 0.8,
        "evidence_sufficient": True,
        "citations": ["obs-1"],
        "abstain_reason": None,
        "recommended_actions": ["roll back auth"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_parses_a_valid_call_tool_decision() -> None:
    decision, error = parse_decision(_call_tool_payload())
    assert error is None
    assert decision is not None
    assert decision.action == "call_tool"
    assert decision.tool == "search_docs"
    assert decision.arguments == {"query": "PAY-50231"}


def test_parses_a_valid_final_answer_decision() -> None:
    decision, error = parse_decision(_final_answer_payload())
    assert error is None
    assert decision is not None
    assert decision.action == "final_answer"
    assert decision.citations == ["obs-1"]


def test_markdown_fenced_json_is_unwrapped() -> None:
    fenced = f"```json\n{_call_tool_payload()}\n```"
    decision, error = parse_decision(fenced)
    assert error is None
    assert decision is not None
    assert decision.tool == "search_docs"


def test_invalid_json_fails_to_parse() -> None:
    decision, error = parse_decision("this is not json at all")
    assert decision is None
    assert error is not None
    assert "parsed" in error


def test_schema_violation_fails_to_parse() -> None:
    decision, error = parse_decision(json.dumps({"action": "not-a-real-action"}))
    assert decision is None
    assert error is not None


def test_call_tool_without_a_tool_name_fails_to_parse() -> None:
    decision, error = parse_decision(_call_tool_payload(tool=None))
    assert decision is None
    assert error is not None
    assert "tool" in error


def test_verify_citations_passes_through_a_real_citation() -> None:
    decision, _ = parse_decision(_final_answer_payload())
    assert decision is not None
    finding = verify_citations(decision, known_observation_ids=KNOWN_OBS)
    assert finding.evidence_sufficient is True
    assert finding.citations == ["obs-1"]


def test_verify_citations_drops_fabricated_and_forces_abstention() -> None:
    decision, _ = parse_decision(_final_answer_payload(citations=["obs-does-not-exist"]))
    assert decision is not None
    finding = verify_citations(decision, known_observation_ids=KNOWN_OBS)
    assert finding.evidence_sufficient is False
    assert finding.citations == []
    assert finding.abstain_reason is not None
    assert "fabricated" in finding.abstain_reason


def test_verify_citations_keeps_only_verified_citations_on_partial_fabrication() -> None:
    decision, _ = parse_decision(_final_answer_payload(citations=["obs-1", "obs-does-not-exist"]))
    assert decision is not None
    finding = verify_citations(decision, known_observation_ids=KNOWN_OBS)
    assert finding.citations == ["obs-1"]
    assert finding.evidence_sufficient is True


def test_verify_citations_leaves_an_honest_abstention_alone() -> None:
    decision, _ = parse_decision(
        _final_answer_payload(
            evidence_sufficient=False, citations=[], abstain_reason="not enough evidence"
        )
    )
    assert decision is not None
    finding = verify_citations(decision, known_observation_ids=KNOWN_OBS)
    assert finding.evidence_sufficient is False
    assert finding.abstain_reason == "not enough evidence"


def test_agent_decision_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(action="final_answer", confidence=1.5)
