"""Enumerated types shared across the data model."""

from __future__ import annotations

import enum


class DocumentType(enum.StrEnum):
    RUNBOOK = "runbook"
    POSTMORTEM = "postmortem"
    ARCHITECTURE = "architecture"
    DEPLOYMENT = "deployment"
    KNOWN_ERROR = "known_error"


class Environment(enum.StrEnum):
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"


class IncidentStatus(enum.StrEnum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Severity(enum.StrEnum):
    SEV1 = "sev1"
    SEV2 = "sev2"
    SEV3 = "sev3"
    SEV4 = "sev4"


class DeploymentStatus(enum.StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class Difficulty(enum.StrEnum):
    STRAIGHTFORWARD = "straightforward"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"
    ADVERSARIAL = "adversarial"


class EvalSplit(enum.StrEnum):
    DEV = "dev"
    TEST = "test"


class RetrievalStrategy(enum.StrEnum):
    """The pipeline an ``EvaluationRun`` measured.

    ``AGENT`` is not a retrieval arm at all — it labels a Phase 8 agent run
    (see ``opspilot.evaluation.agent_runner``) so it sits in the same
    ``evaluation_runs`` table and comparison report as the deterministic
    strategies, per the Phase 9 "does the agent help?" experiment. Never pass
    it to :func:`opspilot.retrieval.strategy.retrieve_candidates` — the
    agent's own tools pick a real strategy (currently always ``HYBRID_RRF``)
    internally.
    """

    VECTOR = "vector"
    LEXICAL = "lexical"
    HYBRID_RRF = "hybrid_rrf"
    AGENT = "agent"

    @classmethod
    def from_name(cls, name: str) -> RetrievalStrategy:
        """Accept the short config/CLI name ``hybrid`` as an alias for ``hybrid_rrf``."""
        if name in ("hybrid", "hybrid_rrf"):
            return cls.HYBRID_RRF
        return cls(name)
