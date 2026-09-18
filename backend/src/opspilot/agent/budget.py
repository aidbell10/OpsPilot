"""Hard budgets on an agent investigation.

Every field here *bounds* the investigation rather than merely guiding it —
the graph checks :meth:`BudgetTracker.exhausted` before every LLM call and
forces a final answer (or abstention) once any limit is hit, so a chatty or
looping LLM can never run away. This is what "no shell, no arbitrary SQL, no
deploys" needs on the cost/time side: a ceiling that holds regardless of what
the model asks for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from opspilot.telemetry.cost import CostAccumulator


@dataclass(frozen=True, slots=True)
class AgentBudget:
    max_tool_calls: int = 6
    max_cost_usd: float = 0.50
    max_seconds: float = 60.0


@dataclass(slots=True)
class BudgetTracker:
    """Mutable running tally for a single investigation."""

    budget: AgentBudget
    cost: CostAccumulator = field(default_factory=CostAccumulator)
    tool_calls_used: int = 0
    _started_at: float = field(default_factory=time.monotonic)

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def record_llm(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.cost.add_llm(model, input_tokens, output_tokens)

    def record_embedding(self, model: str, tokens: int) -> None:
        self.cost.add_embedding(model, tokens)

    def record_tool_call(self) -> None:
        self.tool_calls_used += 1

    def exhausted(self) -> str | None:
        """The first exceeded limit's reason, or ``None`` if still within budget."""
        if self.tool_calls_used >= self.budget.max_tool_calls:
            return f"max_tool_calls ({self.budget.max_tool_calls}) reached"
        if self.elapsed_seconds() >= self.budget.max_seconds:
            return f"max_seconds ({self.budget.max_seconds}) reached"
        if self.cost.total_usd() >= self.budget.max_cost_usd:
            return f"max_cost_usd ({self.budget.max_cost_usd}) reached"
        return None

    def remaining_tool_calls(self) -> int:
        return max(0, self.budget.max_tool_calls - self.tool_calls_used)
