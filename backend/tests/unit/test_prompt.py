from __future__ import annotations

import uuid

import pytest

from opspilot.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from opspilot.models.enums import DocumentType
from opspilot.retrieval.semantic import ChunkMatch

pytestmark = pytest.mark.unit


def _chunk(chunk_id: str = "c1") -> ChunkMatch:
    return ChunkMatch(
        chunk_id=uuid.UUID(int=1),
        document_id=uuid.UUID(int=2),
        content="If checkout returns HTTP 500, check promotion validation.",
        score=0.9,
        title="Checkout troubleshooting",
        source_path="runbooks/checkout/500s.md",
        document_type=DocumentType.RUNBOOK,
        service_name="checkout",
        version="v2.14.0",
    )


def test_system_prompt_frames_evidence_as_untrusted_data() -> None:
    assert "DATA" in SYSTEM_PROMPT
    assert "never" in SYSTEM_PROMPT.lower()
    assert "evidence_sufficient" in SYSTEM_PROMPT


def test_user_prompt_includes_chunk_id_and_content() -> None:
    chunk = _chunk()
    prompt = build_user_prompt(
        description="checkout is returning 500s",
        service="checkout",
        version="v2.14.0",
        environment="production",
        evidence=[chunk],
        related_incidents=[],
    )
    assert str(chunk.chunk_id) in prompt
    assert "promotion validation" in prompt
    assert "checkout is returning 500s" in prompt


def test_user_prompt_handles_no_evidence() -> None:
    prompt = build_user_prompt(
        description="something broke",
        service=None,
        version=None,
        environment=None,
        evidence=[],
        related_incidents=[],
    )
    assert "no evidence was retrieved" in prompt
