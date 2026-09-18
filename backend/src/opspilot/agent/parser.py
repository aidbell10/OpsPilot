"""Parse and validate the LLM's per-turn decision, and verify final citations.

Mirrors ``opspilot.generation.parser``'s philosophy exactly, generalized from
one-shot chunk citations to a multi-step observation trail: malformed output
never raises, and an uncited "sufficient evidence" claim is downgraded to an
abstention rather than trusted.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from opspilot.schemas.agent import AgentDecision, AgentFinding

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    return text


def parse_decision(raw_text: str) -> tuple[AgentDecision | None, str | None]:
    """Parse ``raw_text`` as an :class:`AgentDecision`.

    Returns ``(decision, None)`` on success or ``(None, reason)`` on failure —
    exactly one is non-``None``. A parse failure here means the model isn't
    following the protocol at all (not even a recoverable "wrong tool name"
    mistake), so callers should treat it as an immediate abstention rather
    than retry.
    """
    candidate = _strip_code_fence(raw_text)
    try:
        payload = json.loads(candidate)
        decision = AgentDecision.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, f"decision output could not be parsed as valid structured JSON: {exc}"

    if decision.action == "call_tool" and not decision.tool:
        return None, "action was 'call_tool' but no tool name was given"
    return decision, None


def verify_citations(decision: AgentDecision, *, known_observation_ids: set[str]) -> AgentFinding:
    """Build the final :class:`AgentFinding`, dropping any citation that isn't a real observation.

    If dropping citations leaves a claimed-sufficient result with no support
    at all, the result is downgraded to an abstention — never trust an
    uncited claim, same rule as ``generation.parser.parse_llm_output``.
    """
    verified = [c for c in decision.citations if c in known_observation_ids]
    dropped = len(decision.citations) - len(verified)

    if decision.evidence_sufficient and not verified:
        suffix = f" ({dropped} fabricated citation(s) dropped)" if dropped else ""
        return AgentFinding(
            hypothesis=decision.hypothesis,
            root_cause=decision.root_cause,
            confidence=decision.confidence,
            evidence_sufficient=False,
            citations=[],
            abstain_reason=(
                "model claimed sufficient evidence but no citation matched a known "
                f"observation id{suffix}"
            ),
            recommended_actions=decision.recommended_actions,
        )

    finding = AgentFinding.from_decision(decision)
    if dropped:
        finding = finding.model_copy(update={"citations": verified})
    return finding
