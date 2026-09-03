"""The planted ground truth is what Phase 4 measures retrieval/generation against."""

from __future__ import annotations

import pytest

from data.generator.spec import GeneratorSpec
from data.generator.world import World, build_world


@pytest.fixture(scope="module")
def world() -> World:
    return build_world(GeneratorSpec(seed=42))


def test_answerable_cases_point_at_real_evidence(world: World) -> None:
    by_path = {d.source_path: d for d in world.documents}
    answerable = [c for c in world.ground_truth if c.answerable]
    assert len(answerable) >= 12

    for case in answerable:
        assert case.expected_root_cause
        assert case.required_document_source_paths
        for path in case.required_document_source_paths:
            assert path in by_path, f"{case.case_id}: {path} missing"
        for snippet in case.required_evidence_snippets:
            assert any(
                snippet in by_path[p].content for p in case.required_document_source_paths
            ), f"{case.case_id}: {snippet!r} not in any required doc"


def test_difficulty_mix_is_balanced(world: World) -> None:
    by_diff: dict[str, int] = {}
    for c in world.ground_truth:
        by_diff[c.difficulty] = by_diff.get(c.difficulty, 0) + 1
    assert by_diff.get("straightforward", 0) >= 3
    assert by_diff.get("multi_hop", 0) >= 6
    assert by_diff.get("adversarial", 0) >= 1
    assert by_diff.get("unanswerable", 0) >= 3


def test_every_chain_has_at_least_one_case(world: World) -> None:
    for chain in world.chains:
        assert [c for c in world.ground_truth if c.chain_id == chain.chain_id], chain.chain_id


def test_unanswerable_cases_expect_abstention(world: World) -> None:
    unanswerable = [c for c in world.ground_truth if not c.answerable]
    assert len(unanswerable) >= 3
    for case in unanswerable:
        assert case.difficulty == "unanswerable"
        assert not case.required_document_source_paths
        assert case.expected_root_cause is None
        assert case.acceptable_actions  # "state that evidence is insufficient", ...


def test_adversarial_cases_forbid_the_injected_actions(world: World) -> None:
    adversarial = [c for c in world.ground_truth if c.difficulty == "adversarial"]
    assert adversarial
    for case in adversarial:
        joined = " ".join(case.forbidden_claims).lower()
        assert "database" in joined or "delete" in joined or "drop" in joined


def test_error_codes_are_service_namespaced_and_unique(world: World) -> None:
    codes = [c.error_code for c in world.chains]
    assert len(codes) == len(set(codes))
    prefixes = {c.service_name: c.error_code.split("-")[0] for c in world.chains}
    assert prefixes["payments"] == "PAY"
    assert prefixes["auth"] == "AUTH"


def test_lexical_signal_present_error_code_appears_verbatim(world: World) -> None:
    """A straightforward case must have its error code in a runbook or known-error
    doc so lexical retrieval can win outright (the Phase 5 experiment)."""
    docs = world.documents
    for chain in world.chains:
        if chain.difficulty != "straightforward":
            continue
        hits = [
            d.source_path
            for d in docs
            if d.document_type in {"runbook", "known_error"} and chain.error_code in d.content
        ]
        assert hits, chain.chain_id
