"""Token-bounded chunking with configurable overlap.

Uses ``tiktoken``'s ``cl100k_base`` encoding as a stable, fast tokenizer proxy
— it does not need to match any particular LLM's exact tokenizer, only to
give ``chunk_size``/``chunk_overlap`` a consistent, model-independent unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    content: str
    token_count: int


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split ``text`` into overlapping windows of at most ``chunk_size`` tokens.

    ``chunk_overlap`` must be smaller than ``chunk_size`` (enforced by
    :class:`opspilot.config.Settings`). Returns ``[]`` for blank input.
    """
    if not text.strip():
        return []

    enc = _encoding()
    tokens = enc.encode(text)
    if not tokens:
        return []

    stride = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    start = 0
    while True:
        window = tokens[start : start + chunk_size]
        content = enc.decode(window).strip()
        if content:
            chunks.append(Chunk(index=len(chunks), content=content, token_count=len(window)))
        if start + chunk_size >= len(tokens):
            break
        start += stride
    return chunks
