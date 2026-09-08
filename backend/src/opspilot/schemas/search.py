"""``POST /search`` response shape."""

from __future__ import annotations

from pydantic import BaseModel

from opspilot.schemas.analysis import EvidenceItem


class SearchResponse(BaseModel):
    query: str
    strategy: str
    reranker: str | None = None
    results: list[EvidenceItem]
