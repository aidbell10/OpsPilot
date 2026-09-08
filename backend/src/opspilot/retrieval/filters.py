"""Metadata filter predicates shared by every retrieval arm.

A retrieval query can be scoped by service, document type, version,
environment, and a chunk-timestamp window. The same :class:`ChunkFilters`
object is applied identically to the vector
(:mod:`opspilot.retrieval.semantic`) and lexical
(:mod:`opspilot.retrieval.lexical`) ``SELECT``\\ s — and therefore to their RRF
fusion — so a filter can never mean one thing to one arm and something else to
the other.

The predicates are plain SQL over columns denormalised onto ``document_chunks``
at ingestion time (see :class:`opspilot.models.document.DocumentChunk`), so no
join is ever needed to filter.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import ColumnElement

from opspilot.models.document import DocumentChunk
from opspilot.models.enums import DocumentType, Environment


@dataclass(frozen=True, slots=True)
class ChunkFilters:
    """Optional metadata predicates for a chunk retrieval query.

    Every field defaults to ``None`` (no constraint). ``timestamp_after`` /
    ``timestamp_before`` bound ``DocumentChunk.chunk_timestamp`` inclusively.
    """

    service_name: str | None = None
    document_type: DocumentType | None = None
    version: str | None = None
    environment: Environment | None = None
    timestamp_after: dt.datetime | None = None
    timestamp_before: dt.datetime | None = None

    @property
    def is_empty(self) -> bool:
        return all(
            value is None
            for value in (
                self.service_name,
                self.document_type,
                self.version,
                self.environment,
                self.timestamp_after,
                self.timestamp_before,
            )
        )

    def clauses(self) -> list[ColumnElement[bool]]:
        """Return the filter as a list of AND-combined SQL boolean expressions."""
        out: list[ColumnElement[bool]] = []
        if self.service_name is not None:
            out.append(DocumentChunk.service_name == self.service_name)
        if self.document_type is not None:
            out.append(DocumentChunk.document_type == self.document_type)
        if self.version is not None:
            out.append(DocumentChunk.version == self.version)
        if self.environment is not None:
            out.append(DocumentChunk.environment == self.environment)
        if self.timestamp_after is not None:
            out.append(DocumentChunk.chunk_timestamp >= self.timestamp_after)
        if self.timestamp_before is not None:
            out.append(DocumentChunk.chunk_timestamp <= self.timestamp_before)
        return out
