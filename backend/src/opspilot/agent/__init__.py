"""Agentic investigation (Phase 8).

A LangGraph orchestrator with narrow, read-only, strongly-typed tools
(search_docs, find_similar_incidents, get_incident_logs, get_deployment,
get_service_dependencies, query_metrics) and hard budgets: max tool calls,
max investigation cost, timeouts, retry limits, token caps, explicit error
states. The agent may never mutate infrastructure or run arbitrary code.
"""
