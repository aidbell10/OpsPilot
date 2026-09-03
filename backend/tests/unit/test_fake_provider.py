from __future__ import annotations

import math

import pytest

from opspilot.providers.fake import FakeEmbeddingProvider, FakeLLMProvider


@pytest.mark.unit
def test_fake_embeddings_are_deterministic() -> None:
    provider = FakeEmbeddingProvider(dim=384)
    a = provider.embed(["checkout returns HTTP 500 after v2.14.0"])
    b = provider.embed(["checkout returns HTTP 500 after v2.14.0"])
    assert a.vectors == b.vectors
    assert a.dim == 384


@pytest.mark.unit
def test_fake_embeddings_are_unit_norm() -> None:
    provider = FakeEmbeddingProvider(dim=128)
    (vec,) = provider.embed(["some incident text"]).vectors
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-6)


@pytest.mark.unit
def test_fake_embeddings_reflect_lexical_overlap() -> None:
    provider = FakeEmbeddingProvider(dim=256)
    base = provider.embed(["promotion validation error in checkout service"]).vectors[0]
    near = provider.embed(["promotion validation error in checkout"]).vectors[0]
    far = provider.embed(["unrelated notification email latency spike"]).vectors[0]

    def dot(u: list[float], v: list[float]) -> float:
        return sum(x * y for x, y in zip(u, v, strict=True))

    assert dot(base, near) > dot(base, far)


@pytest.mark.unit
def test_fake_llm_is_deterministic_and_labelled() -> None:
    llm = FakeLLMProvider()
    r1 = llm.complete(system="s", user="u", max_tokens=64)
    r2 = llm.complete(system="s", user="u", max_tokens=64)
    assert r1.text == r2.text
    assert r1.text.startswith("[fake-llm]")
    assert r1.output_tokens > 0
