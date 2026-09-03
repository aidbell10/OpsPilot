"""Structural checks on the generated world."""

from __future__ import annotations

import pytest

from data.generator.company import SERVICE_NAMES
from data.generator.emit import verify
from data.generator.spec import GeneratorSpec
from data.generator.world import World, build_world


@pytest.fixture(scope="module")
def world() -> World:
    return build_world(GeneratorSpec(seed=42))


def test_verify_passes(world: World) -> None:
    verify(world)  # raises on any broken invariant


def test_corpus_is_a_reasonable_size(world: World) -> None:
    counts = world.counts()
    assert counts["services"] == len(SERVICE_NAMES)
    assert counts["documents"] >= 70
    assert counts["documents.runbook"] >= 18
    assert counts["documents.postmortem"] >= 10
    assert counts["documents.architecture"] >= 12
    assert counts["documents.known_error"] >= 10
    assert 20 <= counts["historical_incidents"] <= 35
    assert counts["chains"] == 12


def test_every_service_has_an_architecture_doc(world: World) -> None:
    have = {
        d.service_name
        for d in world.documents
        if d.document_type == "architecture" and d.service_name is not None
    }
    assert set(SERVICE_NAMES) <= have


def test_deployments_are_unique_and_ordered(world: World) -> None:
    seen: set[tuple[str, str]] = set()
    last_by_service: dict[str, str] = {}
    for d in world.deployments:
        key = (d.service_name, d.version)
        assert key not in seen
        seen.add(key)
        prev = last_by_service.get(d.service_name)
        if prev is not None:
            assert d.deployed_at >= prev
        last_by_service[d.service_name] = d.deployed_at


def test_planted_trigger_and_fix_releases_exist(world: World) -> None:
    triggers = [d for d in world.deployments if d.is_planted_trigger]
    fixes = [d for d in world.deployments if d.is_planted_fix]
    assert len(triggers) == 12
    assert len(fixes) == 12
