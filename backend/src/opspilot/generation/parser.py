"""Parse, validate, and citation-check raw LLM text into an ``AnalysisResult``.

Never raises on malformed model output — a JSON parse failure or a schema
validation failure becomes a deterministic abstained result instead, so
``POST /incidents/analyze`` always returns 200 even when the configured LLM
provider produces garbage (as the offline ``fake`` provider deliberately
does).
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from opspilot.providers.base import LLMProvider
from opspilot.schemas.analysis import AnalysisResult

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    return text


def _abstain(reason: str) -> AnalysisResult:
    return AnalysisResult(
        hypothesis="",
        root_cause=None,
        confidence=0.0,
        evidence_sufficient=False,
        citations=[],
        abstain_reason=reason,
    )


def parse_llm_output(raw_text: str, *, valid_chunk_ids: set[str]) -> AnalysisResult:
    """Parse ``raw_text`` as :class:`AnalysisResult`, verifying every citation.

    Citations that don't match a retrieved chunk id are dropped, never
    trusted. If dropping citations leaves a claimed-sufficient result with no
    supporting evidence at all, the result is downgraded to an abstention.
    """
    candidate = _strip_code_fence(raw_text)
    try:
        payload = json.loads(candidate)
        result = AnalysisResult.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        return _abstain(f"generation output could not be parsed as valid structured JSON: {exc}")

    verified = [c for c in result.citations if c in valid_chunk_ids]
    dropped = len(result.citations) - len(verified)

    if result.evidence_sufficient and not verified:
        suffix = f" ({dropped} fabricated citation(s) dropped)" if dropped else ""
        return result.model_copy(
            update={
                "evidence_sufficient": False,
                "citations": [],
                "abstain_reason": (
                    "model claimed sufficient evidence but no citation matched a "
                    f"retrieved chunk id{suffix}"
                ),
            }
        )

    if dropped:
        return result.model_copy(update={"citations": verified})
    return result


def run_generation(
    llm: LLMProvider,
    *,
    system: str,
    user: str,
    valid_chunk_ids: set[str],
    max_tokens: int,
) -> AnalysisResult:
    completion = llm.complete(system=system, user=user, max_tokens=max_tokens)
    return parse_llm_output(completion.text, valid_chunk_ids=valid_chunk_ids)
