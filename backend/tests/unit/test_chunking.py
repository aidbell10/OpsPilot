from __future__ import annotations

import pytest

from opspilot.ingestion.chunking import chunk_text

pytestmark = pytest.mark.unit


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("", chunk_size=64, chunk_overlap=8) == []
    assert chunk_text("   \n\t  ", chunk_size=64, chunk_overlap=8) == []


def test_short_text_is_a_single_chunk() -> None:
    chunks = chunk_text("a short sentence about checkout errors", chunk_size=64, chunk_overlap=8)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].token_count <= 64
    assert "checkout" in chunks[0].content


def test_long_text_is_split_with_overlap() -> None:
    text = " ".join(f"token{i}" for i in range(400))
    chunks = chunk_text(text, chunk_size=64, chunk_overlap=16)

    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.token_count <= 64 for c in chunks)
    # consecutive chunks overlap: the tail of one reappears at the head of the next
    assert chunks[0].content.split()[-1] in chunks[1].content.split()[:20]


def test_no_overlap_chunks_do_not_repeat_tokens() -> None:
    text = " ".join(f"tok{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=32, chunk_overlap=0)

    seen: set[str] = set()
    for chunk in chunks:
        words = set(chunk.content.split())
        assert not (seen & words), "zero-overlap chunks must not repeat tokens"
        seen |= words


def test_handles_unicode() -> None:
    text = "café résumé naïve café résumé naïve " * 20
    chunks = chunk_text(text, chunk_size=16, chunk_overlap=4)
    assert chunks
    assert all(c.content for c in chunks)


def test_reconstructs_full_text_when_unchunked() -> None:
    text = "checkout returns HTTP 500 after deployment v2.14.0"
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)
    assert len(chunks) == 1
    assert chunks[0].content == text
