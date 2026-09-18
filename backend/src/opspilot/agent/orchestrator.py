"""Top-level entry point: ``investigate(...)`` runs one bounded agent investigation.

The one function the API route (and, later, Phase 9's evaluation harness)
calls — builds a fresh graph and budget tracker per call, so investigations
never share state, then unpacks the final :class:`~opspilot.agent.graph.AgentState`
into a plain result the caller doesn't need to know is LangGraph-shaped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sqlalchemy.orm import Session

from opspilot.agent.budget import AgentBudget, BudgetTracker
from opspilot.agent.graph import AgentState, build_graph
from opspilot.agent.tools import ToolContext
from opspilot.config import Settings
from opspilot.providers.base import EmbeddingProvider, LLMProvider, RerankProvider
from opspilot.schemas.agent import AgentDecision, AgentFinding, Observation

# Generous margin over 2 decide/act pairs per tool call + the final decide —
# a backstop against a graph bug, not the real limit (BudgetTracker is).
_RECURSION_MARGIN = 6


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    finding: AgentFinding
    observations: list[Observation]
    decision_log: list[AgentDecision]
    stop_reason: str
    tool_calls_used: int
    elapsed_seconds: float
    cost: dict[str, float | str]


def investigate(
    session: Session,
    *,
    description: str,
    service_name: str | None,
    version: str | None,
    environment: str | None,
    llm_provider: LLMProvider,
    embedding_provider: EmbeddingProvider,
    rerank_provider: RerankProvider | None,
    settings: Settings,
    budget: AgentBudget | None = None,
) -> AgentRunResult:
    budget = budget or AgentBudget(
        max_tool_calls=settings.agent_max_tool_calls,
        max_cost_usd=settings.agent_max_cost_usd,
        max_seconds=settings.agent_max_seconds,
    )
    tracker = BudgetTracker(budget=budget)
    tool_ctx = ToolContext(
        embedding_provider=embedding_provider,
        rerank_provider=rerank_provider,
        settings=settings,
        cost=tracker.cost,
    )
    graph = build_graph(
        session=session, llm_provider=llm_provider, tool_ctx=tool_ctx, tracker=tracker
    )

    initial_state: AgentState = {
        "description": description,
        "service_name": service_name,
        "version": version,
        "environment": environment,
        "observations": [],
        "decision_log": [],
        "pending_tool_call": None,
        "finding": None,
        "stop_reason": None,
    }
    result = cast(
        AgentState,
        graph.invoke(
            initial_state,
            config={"recursion_limit": 2 * budget.max_tool_calls + _RECURSION_MARGIN},
        ),
    )

    return AgentRunResult(
        finding=result["finding"] or AgentFinding.abstain("agent produced no finding"),
        observations=result["observations"],
        decision_log=result["decision_log"],
        stop_reason=result["stop_reason"] or "unknown",
        tool_calls_used=tracker.tool_calls_used,
        elapsed_seconds=tracker.elapsed_seconds(),
        cost=tracker.cost.as_dict(),
    )
