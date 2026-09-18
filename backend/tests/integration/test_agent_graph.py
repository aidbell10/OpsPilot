"""Phase 8 — the full decide/act loop against a live DB, with a scripted LLM.

A scripted stub (not the real ``fake`` provider, which always returns garbage
on purpose) lets these tests drive specific multi-step conversations: a
normal tool-call-then-answer loop, a model that never stops calling tools
(budget must force it to abstain), and a model that asks for a nonexistent
tool (a recoverable mistake, unlike unparseable JSON).
"""

from __future__ import annotations

import json

import pytest
from corpus_fixtures import agent_corpus
from sqlalchemy.orm import Session

from opspilot.agent.budget import AgentBudget
from opspilot.agent.orchestrator import investigate
from opspilot.config import get_settings
from opspilot.ingestion.pipeline import ingest_corpus
from opspilot.providers.base import LLMResult
from opspilot.providers.fake import FakeEmbeddingProvider, FakeLLMProvider

pytestmark = [pytest.mark.integration, pytest.mark.agent]


class _ScriptedLLM:
    """Returns queued canned responses in order, ignoring the actual prompt."""

    name = "scripted"
    model = "scripted-v1"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts_seen: list[str] = []

    def complete(
        self, *, system: str, user: str, max_tokens: int, temperature: float = 0.0
    ) -> LLMResult:
        self.prompts_seen.append(user)
        text = (
            self._responses.pop(0)
            if self._responses
            else json.dumps({"action": "final_answer", "evidence_sufficient": False})
        )
        return LLMResult(text=text, model=self.model, input_tokens=5, output_tokens=5)


def _call_tool(tool: str, arguments: dict[str, object]) -> str:
    return json.dumps({"action": "call_tool", "tool": tool, "arguments": arguments})


def _final_answer(**overrides: object) -> str:
    payload: dict[str, object] = {
        "action": "final_answer",
        "hypothesis": "promotion validation bug",
        "confidence": 0.9,
        "evidence_sufficient": True,
        "citations": ["obs-1"],
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.fixture
def seeded(db_session: Session, clean_db: None) -> Session:
    ingest_corpus(
        db_session,
        agent_corpus(),
        embedding_provider=FakeEmbeddingProvider(dim=384),
        chunk_size=48,
        chunk_overlap=8,
    )
    db_session.commit()
    return db_session


def _investigate(session: Session, llm: object, **kwargs: object) -> object:
    return investigate(
        session,
        description="checkout is returning HTTP 500 for carts with a promotion applied",
        service_name="checkout",
        version=None,
        environment=None,
        llm_provider=llm,  # type: ignore[arg-type]
        embedding_provider=FakeEmbeddingProvider(dim=384),
        rerank_provider=None,
        settings=get_settings(),
        **kwargs,
    )


def test_happy_path_calls_a_tool_then_answers_with_a_verified_citation(
    seeded: Session,
) -> None:
    llm = _ScriptedLLM(
        [
            _call_tool("search_docs", {"query": "checkout promotion validate_cart"}),
            _final_answer(),
        ]
    )
    run = _investigate(seeded, llm)

    assert run.stop_reason == "final_answer"
    assert run.tool_calls_used == 1
    assert len(run.observations) == 1
    assert run.observations[0].ok is True
    assert run.finding.evidence_sufficient is True
    assert run.finding.citations == ["obs-1"]


def test_fabricated_citation_is_downgraded_to_abstention(seeded: Session) -> None:
    llm = _ScriptedLLM(
        [
            _call_tool("search_docs", {"query": "checkout"}),
            _final_answer(citations=["obs-does-not-exist"]),
        ]
    )
    run = _investigate(seeded, llm)

    assert run.finding.evidence_sufficient is False
    assert run.finding.abstain_reason is not None
    assert "fabricated" in run.finding.abstain_reason
    assert run.finding.citations == []


def test_budget_forces_abstention_when_the_model_never_stops_calling_tools(
    seeded: Session,
) -> None:
    llm = _ScriptedLLM(
        [
            _call_tool("search_docs", {"query": "checkout 1"}),
            _call_tool("search_docs", {"query": "checkout 2"}),  # forced-final turn; disobeys
            _call_tool("search_docs", {"query": "checkout 3"}),
        ]
    )
    run = _investigate(seeded, llm, budget=AgentBudget(max_tool_calls=1))

    assert run.tool_calls_used == 1  # act() never runs a second time
    assert run.stop_reason == "budget_exhausted"
    assert run.finding.evidence_sufficient is False
    assert run.finding.abstain_reason is not None
    assert "budget" in run.finding.abstain_reason.lower()


def test_unknown_tool_name_is_recorded_and_the_model_gets_another_turn(
    seeded: Session,
) -> None:
    llm = _ScriptedLLM(
        [
            _call_tool("delete_everything", {}),
            _final_answer(evidence_sufficient=False, citations=[], abstain_reason="no evidence"),
        ]
    )
    run = _investigate(seeded, llm)

    assert run.tool_calls_used == 1
    assert run.observations[0].ok is False
    assert run.observations[0].error is not None
    assert "unknown tool" in run.observations[0].error
    assert run.stop_reason == "final_answer"
    assert run.finding.evidence_sufficient is False


def test_unparseable_llm_output_aborts_immediately_with_zero_tool_calls(
    seeded: Session,
) -> None:
    # The real FakeLLMProvider deliberately never emits JSON.
    run = _investigate(seeded, FakeLLMProvider())

    assert run.tool_calls_used == 0
    assert run.stop_reason == "parse_error"
    assert run.finding.evidence_sufficient is False
    assert run.finding.abstain_reason is not None


def test_recommended_actions_never_claim_to_have_been_performed(seeded: Session) -> None:
    llm = _ScriptedLLM(
        [
            _call_tool("search_docs", {"query": "checkout"}),
            _final_answer(recommended_actions=["roll back checkout to v2.13.0"]),
        ]
    )
    run = _investigate(seeded, llm)
    assert run.finding.recommended_actions == ["roll back checkout to v2.13.0"]
