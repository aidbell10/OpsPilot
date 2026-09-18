"""Structured contract for the LangGraph agent's decision loop (Phase 8).

One model, two shapes, discriminated by ``action`` — mirrors
``opspilot.schemas.analysis.AnalysisResult``'s "one schema drives both the
prompt and the validation" approach, generalized to a multi-step loop: at
each iteration the LLM emits an :class:`AgentDecision` that is either another
tool call or a final answer. Never trusted uncited, same as Phase 3.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentDecision(BaseModel):
    """What the LLM wants to do next, or its final finding."""

    action: Literal["call_tool", "final_answer"]
    thought: str = Field(default="", description="One sentence: why this action, now.")

    # Populated when action == "call_tool".
    tool: str | None = Field(default=None, description="Name of the tool to call.")
    arguments: dict[str, object] = Field(
        default_factory=dict, description="Arguments for the tool call."
    )

    # Populated when action == "final_answer" (same shape as AnalysisResult).
    hypothesis: str = Field(default="", description="A one- or two-sentence root-cause hypothesis.")
    root_cause: str | None = Field(default=None, description="Detailed root-cause explanation.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_sufficient: bool = Field(
        default=False, description="True only if the cited observations support the hypothesis."
    )
    citations: list[str] = Field(
        default_factory=list, description="Observation ids (e.g. 'obs-2') that support the finding."
    )
    abstain_reason: str | None = Field(
        default=None, description="Required when evidence_sufficient is false."
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Suggested next steps for a human — never claim to have performed them.",
    )


class AgentFinding(BaseModel):
    """The agent's final, citation-verified result — returned to the caller."""

    hypothesis: str = ""
    root_cause: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_sufficient: bool = False
    citations: list[str] = Field(default_factory=list)
    abstain_reason: str | None = None
    recommended_actions: list[str] = Field(default_factory=list)

    @classmethod
    def from_decision(cls, decision: AgentDecision) -> AgentFinding:
        return cls(
            hypothesis=decision.hypothesis,
            root_cause=decision.root_cause,
            confidence=decision.confidence,
            evidence_sufficient=decision.evidence_sufficient,
            citations=decision.citations,
            abstain_reason=decision.abstain_reason,
            recommended_actions=decision.recommended_actions,
        )

    @classmethod
    def abstain(cls, reason: str) -> AgentFinding:
        return cls(evidence_sufficient=False, abstain_reason=reason)


class Observation(BaseModel):
    """One tool call's recorded result — DATA the LLM reads, never instructions.

    ``id`` is what a final answer's ``citations`` must reference; ``content``
    is the human-readable rendering shown in the prompt.
    """

    id: str
    tool: str
    arguments: dict[str, object]
    ok: bool
    content: str
    error: str | None = None


class InvestigateResponse(BaseModel):
    """``POST /incidents/investigate``'s response — the agent's full, auditable run."""

    incident_id: str
    finding: AgentFinding
    observations: list[Observation]
    decision_log: list[AgentDecision]
    stop_reason: str
    tool_calls_used: int
    elapsed_seconds: float
    cost: dict[str, float | str]
