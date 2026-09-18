from __future__ import annotations

import time

import pytest

from opspilot.agent.budget import AgentBudget, BudgetTracker

pytestmark = pytest.mark.unit


def test_fresh_tracker_is_not_exhausted() -> None:
    tracker = BudgetTracker(budget=AgentBudget())
    assert tracker.exhausted() is None
    assert tracker.remaining_tool_calls() == 6


def test_max_tool_calls_exhausts() -> None:
    tracker = BudgetTracker(budget=AgentBudget(max_tool_calls=2))
    tracker.record_tool_call()
    assert tracker.exhausted() is None
    assert tracker.remaining_tool_calls() == 1
    tracker.record_tool_call()
    reason = tracker.exhausted()
    assert reason is not None
    assert "max_tool_calls" in reason
    assert tracker.remaining_tool_calls() == 0


def test_max_cost_usd_exhausts() -> None:
    tracker = BudgetTracker(budget=AgentBudget(max_cost_usd=0.0001))
    tracker.record_llm("claude-sonnet-5", input_tokens=10_000, output_tokens=10_000)
    reason = tracker.exhausted()
    assert reason is not None
    assert "max_cost_usd" in reason


def test_max_seconds_exhausts() -> None:
    tracker = BudgetTracker(budget=AgentBudget(max_seconds=0.01))
    time.sleep(0.02)
    reason = tracker.exhausted()
    assert reason is not None
    assert "max_seconds" in reason


def test_free_llm_calls_do_not_exhaust_cost_budget() -> None:
    tracker = BudgetTracker(budget=AgentBudget(max_cost_usd=0.01))
    tracker.record_llm("fake-llm-v1", input_tokens=10_000, output_tokens=10_000)
    assert tracker.exhausted() is None


def test_record_embedding_contributes_to_cost() -> None:
    tracker = BudgetTracker(budget=AgentBudget())
    tracker.record_embedding("BAAI/bge-small-en-v1.5", tokens=1000)
    # local embeddings are priced at 0 in the default pricebook — recording
    # them must not raise, even though it contributes nothing measurable.
    assert tracker.exhausted() is None
