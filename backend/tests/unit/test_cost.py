from __future__ import annotations

import pytest

from opspilot.telemetry.cost import CostAccumulator, PriceBook, estimate_cost_usd


@pytest.mark.unit
def test_local_and_fake_models_are_free() -> None:
    cost = estimate_cost_usd(
        llm_model="fake-llm-v1",
        llm_input_tokens=1000,
        llm_output_tokens=500,
        embedding_model="fake-embed-v1",
        embedding_tokens=10_000,
    )
    assert cost == 0.0


@pytest.mark.unit
def test_llm_cost_uses_pricebook_rates() -> None:
    pb = PriceBook()
    # claude-sonnet-5: (3.0 in, 15.0 out) per 1M tokens
    cost = pb.llm_cost_usd("claude-sonnet-5", 1_000_000, 1_000_000)
    assert cost == pytest.approx(18.0)


@pytest.mark.unit
def test_accumulator_sums_and_serialises() -> None:
    acc = CostAccumulator()
    acc.add_embedding("BAAI/bge-small-en-v1.5", 4000)
    acc.add_llm("claude-sonnet-5", 2000, 800)
    acc.add_llm("claude-sonnet-5", 1000, 200)

    assert acc.llm_input_tokens == 3000
    assert acc.llm_output_tokens == 1000
    d = acc.as_dict()
    assert d["llm_input_tokens"] == 3000
    assert d["estimated_usd"] == pytest.approx(3000 * 3.0 / 1e6 + 1000 * 15.0 / 1e6)
    assert d["pricebook_version"]


@pytest.mark.unit
def test_unknown_model_costs_zero_not_crash() -> None:
    pb = PriceBook()
    assert pb.llm_cost_usd("some-future-model", 100, 100) == 0.0
