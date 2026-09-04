"""Vector-only evidence search (Phase 3). Hybrid lexical+RRF arrives in Phase 5."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from opspilot.db.session import get_db
from opspilot.models.enums import DocumentType
from opspilot.providers.factory import get_embedding_provider
from opspilot.retrieval.semantic import embed_and_search_chunks
from opspilot.schemas.analysis import EvidenceItem
from opspilot.schemas.search import SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
    service: str | None = None
    document_type: DocumentType | None = None


@router.post("", response_model=SearchResponse)
def search(request: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    embedding_provider = get_embedding_provider()
    matches = embed_and_search_chunks(
        db,
        embedding_provider,
        request.query,
        top_k=request.top_k,
        service_name=request.service,
        document_type=request.document_type,
    )
    return SearchResponse(
        query=request.query,
        results=[
            EvidenceItem(
                chunk_id=str(m.chunk_id),
                document_id=str(m.document_id),
                title=m.title,
                source_path=m.source_path,
                document_type=m.document_type,
                service_name=m.service_name,
                version=m.version,
                content=m.content,
                score=m.score,
            )
            for m in matches
        ],
    )
