from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.slow]

st = pytest.importorskip("sentence_transformers", reason="requires the 'ml' extra")


def test_local_model_produces_expected_dimension() -> None:
    from opspilot.providers.local_embeddings import LocalEmbeddingProvider

    provider = LocalEmbeddingProvider(model="BAAI/bge-small-en-v1.5", dim=384)
    result = provider.embed(["checkout returns HTTP 500 after deployment v2.14.0"])
    assert result.dim == 384
    (vec,) = result.vectors
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-4)


def test_dimension_mismatch_is_rejected() -> None:
    from opspilot.providers.local_embeddings import LocalEmbeddingProvider

    provider = LocalEmbeddingProvider(model="BAAI/bge-small-en-v1.5", dim=999)
    with pytest.raises(ValueError, match="OPSPILOT_EMBEDDING_DIM"):
        provider.embed(["text"])
