"""The whole point of the generator: same seed in, identical corpus out."""

from __future__ import annotations

from data.generator.emit import generate
from data.generator.spec import GeneratorSpec


def test_two_runs_are_byte_identical() -> None:
    a = generate(GeneratorSpec(seed=42))
    b = generate(GeneratorSpec(seed=42))
    assert a == b


def test_manifest_hash_is_stable_across_instances() -> None:
    h1 = generate(GeneratorSpec(seed=42))["manifest.json"]
    h2 = generate(GeneratorSpec(seed=42))["manifest.json"]
    assert h1 == h2


def test_different_seed_changes_the_corpus() -> None:
    a = generate(GeneratorSpec(seed=42))["manifest.json"]
    b = generate(GeneratorSpec(seed=43))["manifest.json"]
    assert a != b


def test_file_set_is_exactly_as_expected() -> None:
    files = set(generate(GeneratorSpec(seed=1)))
    assert files == {
        "services.json",
        "deployments.json",
        "documents.json",
        "historical_incidents.json",
        "ground_truth.json",
        "chains.json",
        "manifest.json",
    }


def test_no_wallclock_leaks_into_output() -> None:
    # A crude guard: the year 2026+ must never appear (the world ends in 2025).
    blob = "".join(generate(GeneratorSpec(seed=42)).values())
    for year in ("2026", "2027", "2028"):
        assert year not in blob
