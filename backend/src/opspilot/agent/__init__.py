"""Agentic investigation (Phase 8).

A LangGraph orchestrator with five narrow, read-only, strongly-typed tools
(search_docs, find_similar_incidents, get_incident_logs, get_deployment,
get_service_dependencies) and hard budgets: max tool calls, max investigation
cost, and a wall-clock timeout, all enforced before every LLM call so a
looping or chatty model can never run away (see ``opspilot.agent.budget``).
``query_metrics`` from the original roadmap sketch is deliberately not
implemented — this project has no metrics/time-series backend, and
fabricating one to give the tool something to return would be exactly the
kind of made-up number this project's evaluation philosophy refuses
(see ``opspilot.agent.tools``). The agent may never mutate infrastructure,
run arbitrary code, or run arbitrary SQL — its only capabilities are the
five read-only ``select``-backed tools in the registry; any remediation is a
recommendation for a human to review, never an action taken.

Entry point: :func:`opspilot.agent.orchestrator.investigate`.
"""

from opspilot.agent.orchestrator import AgentRunResult, investigate

__all__ = ["AgentRunResult", "investigate"]
