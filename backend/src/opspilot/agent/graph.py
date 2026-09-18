"""The LangGraph control flow: decide -> act -> decide -> ... -> final answer.

Two nodes only — ``decide`` (ask the LLM what to do next) and ``act`` (run
the tool it asked for) — looping until the LLM emits ``final_answer`` or the
budget forces one. All state that must survive the loop (observations,
decision log, the eventual finding) lives in the LangGraph state; the DB
session, providers, and the mutable :class:`BudgetTracker` are closed over by
the node functions instead, since they're runtime dependencies, not data to
persist or replay.

Budget enforcement lives entirely in ``decide``: it checks
``tracker.exhausted()`` *before* asking the LLM anything, and if exhausted it
tells the model this is its last turn — if the model still tries to call a
tool anyway, that attempt is overridden into an abstention rather than
honored, so ``act`` can never run once the budget is spent.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session

from opspilot.agent.budget import BudgetTracker
from opspilot.agent.parser import parse_decision, verify_citations
from opspilot.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from opspilot.agent.tools import ToolContext, dispatch
from opspilot.providers.base import LLMProvider
from opspilot.schemas.agent import AgentDecision, AgentFinding, Observation


class PendingToolCall(TypedDict):
    tool: str
    arguments: dict[str, object]


class AgentState(TypedDict):
    description: str
    service_name: str | None
    version: str | None
    environment: str | None
    observations: list[Observation]
    decision_log: list[AgentDecision]
    pending_tool_call: PendingToolCall | None
    finding: AgentFinding | None
    stop_reason: str | None


def build_graph(
    *,
    session: Session,
    llm_provider: LLMProvider,
    tool_ctx: ToolContext,
    tracker: BudgetTracker,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    def decide(state: AgentState) -> dict[str, object]:
        forced_final = tracker.exhausted() is not None
        user_prompt = build_user_prompt(
            description=state["description"],
            service_name=state["service_name"],
            version=state["version"],
            environment=state["environment"],
            observations=state["observations"],
            remaining_tool_calls=tracker.remaining_tool_calls(),
            forced_final=forced_final,
        )
        completion = llm_provider.complete(
            system=SYSTEM_PROMPT, user=user_prompt, max_tokens=tool_ctx.settings.llm_max_tokens
        )
        tracker.record_llm(llm_provider.model, completion.input_tokens, completion.output_tokens)

        decision, error = parse_decision(completion.text)
        if decision is None:
            return {
                "finding": AgentFinding.abstain(
                    f"agent could not produce a valid decision: {error}"
                ),
                "stop_reason": "parse_error",
            }

        decision_log = [*state["decision_log"], decision]

        if forced_final and decision.action == "call_tool":
            return {
                "decision_log": decision_log,
                "finding": AgentFinding.abstain(
                    "budget exhausted; model attempted another tool call instead of a final answer"
                ),
                "stop_reason": "budget_exhausted",
            }

        if decision.action == "final_answer":
            known_ids = {o.id for o in state["observations"]}
            finding = verify_citations(decision, known_observation_ids=known_ids)
            return {"decision_log": decision_log, "finding": finding, "stop_reason": "final_answer"}

        assert decision.tool is not None  # enforced by parse_decision
        return {
            "decision_log": decision_log,
            "pending_tool_call": {"tool": decision.tool, "arguments": decision.arguments},
        }

    def act(state: AgentState) -> dict[str, object]:
        pending = state["pending_tool_call"]
        assert pending is not None
        next_id = f"obs-{len(state['observations']) + 1}"
        ok, content, error = dispatch(pending["tool"], pending["arguments"], session, tool_ctx)
        tracker.record_tool_call()
        observation = Observation(
            id=next_id,
            tool=pending["tool"],
            arguments=pending["arguments"],
            ok=ok,
            content=content,
            error=error,
        )
        return {"observations": [*state["observations"], observation], "pending_tool_call": None}

    def route_after_decide(state: AgentState) -> str:
        return "end" if state.get("finding") is not None else "act"

    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    graph.add_node("decide", decide)
    graph.add_node("act", act)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", route_after_decide, {"act": "act", "end": END})
    graph.add_edge("act", "decide")
    return graph.compile()
