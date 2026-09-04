from __future__ import annotations

import json

import pytest

from opspilot.generation.parser import parse_llm_output, run_generation
from opspilot.providers.base import LLMResult

pytestmark = pytest.mark.unit

VALID_CHUNK_IDS = {"chunk-a", "chunk-b"}


class _StubLLM:
    """A minimal LLMProvider returning a fixed raw text, for parser tests."""

    name = "stub"
    model = "stub-v1"

    def __init__(self, text: str) -> None:
        self._text = text

    def complete(
        self, *, system: str, user: str, max_tokens: int, temperature: float = 0.0
    ) -> LLMResult:
        return LLMResult(text=self._text, model=self.model, input_tokens=1, output_tokens=1)


def _valid_payload(**overrides: object) -> str:
    payload = {
        "hypothesis": "auth JWKS cache went stale after rotate_jwks changed",
        "root_cause": "AUTH_JWKS_CACHE_TTL_S was increased beyond token lifetime",
        "confidence": 0.8,
        "evidence_sufficient": True,
        "citations": ["chunk-a"],
        "abstain_reason": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_valid_json_with_real_citation_passes_through() -> None:
    result = parse_llm_output(_valid_payload(), valid_chunk_ids=VALID_CHUNK_IDS)
    assert result.evidence_sufficient is True
    assert result.citations == ["chunk-a"]
    assert result.confidence == 0.8


def test_markdown_fenced_json_is_unwrapped() -> None:
    fenced = f"```json\n{_valid_payload()}\n```"
    result = parse_llm_output(fenced, valid_chunk_ids=VALID_CHUNK_IDS)
    assert result.evidence_sufficient is True
    assert result.citations == ["chunk-a"]


def test_invalid_json_abstains() -> None:
    result = parse_llm_output("this is not json at all", valid_chunk_ids=VALID_CHUNK_IDS)
    assert result.evidence_sufficient is False
    assert result.citations == []
    assert result.abstain_reason is not None
    assert "parsed" in result.abstain_reason


def test_schema_violation_abstains() -> None:
    bad = json.dumps({"confidence": "not-a-number"})
    result = parse_llm_output(bad, valid_chunk_ids=VALID_CHUNK_IDS)
    assert result.evidence_sufficient is False
    assert result.abstain_reason is not None


def test_fabricated_citation_is_dropped_and_forces_abstention() -> None:
    payload = _valid_payload(citations=["chunk-does-not-exist"])
    result = parse_llm_output(payload, valid_chunk_ids=VALID_CHUNK_IDS)
    assert result.evidence_sufficient is False
    assert result.citations == []
    assert result.abstain_reason is not None
    assert "fabricated" in result.abstain_reason


def test_partial_fabrication_keeps_only_verified_citations() -> None:
    payload = _valid_payload(citations=["chunk-a", "chunk-does-not-exist"])
    result = parse_llm_output(payload, valid_chunk_ids=VALID_CHUNK_IDS)
    assert result.citations == ["chunk-a"]
    assert result.evidence_sufficient is True


def test_abstained_result_with_no_citations_is_left_alone() -> None:
    payload = _valid_payload(
        evidence_sufficient=False, citations=[], abstain_reason="not enough evidence"
    )
    result = parse_llm_output(payload, valid_chunk_ids=VALID_CHUNK_IDS)
    assert result.evidence_sufficient is False
    assert result.abstain_reason == "not enough evidence"


def test_run_generation_calls_provider_and_parses() -> None:
    llm = _StubLLM(_valid_payload())
    result = run_generation(
        llm, system="sys", user="usr", valid_chunk_ids=VALID_CHUNK_IDS, max_tokens=100
    )
    assert result.evidence_sufficient is True


def test_run_generation_with_fake_provider_output_abstains() -> None:
    # The real FakeLLMProvider deliberately returns non-JSON text; the pipeline
    # must degrade gracefully to a 200-with-abstention rather than erroring.
    from opspilot.providers.fake import FakeLLMProvider

    llm = FakeLLMProvider()
    result = run_generation(
        llm, system="sys", user="usr", valid_chunk_ids=VALID_CHUNK_IDS, max_tokens=100
    )
    assert result.evidence_sufficient is False
    assert result.abstain_reason is not None
