from __future__ import annotations

import uuid

import pytest

from opspilot.models.enums import DocumentType
from opspilot.providers.fake import FakeRerankProvider
from opspilot.retrieval.rerank import rerank_chunks
from opspilot.retrieval.semantic import ChunkMatch

pytestmark = pytest.mark.unit


def _chunk(content: str, *, score: float = 0.0) -> ChunkMatch:
    return ChunkMatch(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        score=score,
        title="t",
        source_path="p",
        document_type=DocumentType.RUNBOOK,
        service_name="checkout",
        version=None,
    )


def test_fake_reranker_is_deterministic_and_bounded() -> None:
    provider = FakeRerankProvider()
    r1 = provider.rerank("promotion validation error", ["promotion validation failed", "cpu graph"])
    r2 = provider.rerank("promotion validation error", ["promotion validation failed", "cpu graph"])
    assert r1.scores == r2.scores
    assert all(0.0 <= s <= 1.0 for s in r1.scores)
    assert r1.scores[0] > r1.scores[1]  # first doc shares more tokens with the query


def test_rerank_chunks_reorders_by_relevance_and_trims() -> None:
    chunks = [
        _chunk("nightly restock job reconciles warehouse counts"),
        _chunk("promotion code validation raises on empty discount"),
        _chunk("checkout returns http 500 after promotion deploy"),
    ]
    out = rerank_chunks(FakeRerankProvider(), "promotion validation http 500", chunks, top_k=2)
    assert len(out) == 2
    # the two promotion/500 chunks outrank the restock chunk
    assert all("restock" not in c.content for c in out)
    # score is now the reranker's score, descending
    assert out[0].score >= out[1].score


def test_rerank_chunks_empty_input() -> None:
    assert rerank_chunks(FakeRerankProvider(), "q", [], top_k=5) == []


def test_rerank_chunks_stable_on_score_ties() -> None:
    # a reranker that can't tell them apart must preserve input order
    class _FlatReranker:
        name = "flat"
        model = "flat"

        def rerank(self, query: str, documents: list[str]):  # type: ignore[no-untyped-def]
            from opspilot.providers.base import RerankResult

            return RerankResult(scores=[0.5] * len(documents), model=self.model)

    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    out = rerank_chunks(_FlatReranker(), "q", chunks, top_k=3)
    assert [c.chunk_id for c in out] == [c.chunk_id for c in chunks]
