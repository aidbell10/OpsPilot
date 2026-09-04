"""``POST /search`` response shape."""

from __future__ import annotations

from pydantic import BaseModel

from opspilot.schemas.analysis import EvidenceItem


class SearchResponse(BaseModel):
    query: str
    results: list[EvidenceItem]
