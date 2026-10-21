from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus


class DocumentProcessing(NamedTuple):
    """Where a document stands in the pipeline, as the inspection view shows it."""

    document_id: int
    name: str
    media_type: MediaType
    status: ProcessingStatus
    error: str
    page_count: int | None
    chunking_version: str
    processed_at: str
    report: dict[str, Any]
    # The indexing stage: status, the model the embeddings came from, its error,
    # and what the last run did (counts and durations only).
    index_status: IndexStatus
    index_model: str
    index_error: str
    indexed_at: str
    index_report: dict[str, Any]
    chunk_count: int
    indexed_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "name": self.name,
            "media_type": self.media_type.value,
            "status": self.status.value,
            "error": self.error,
            "page_count": self.page_count,
            "chunking_version": self.chunking_version,
            "processed_at": self.processed_at,
            "report": self.report,
            "index_status": self.index_status.value,
            "index_model": self.index_model,
            "index_error": self.index_error,
            "indexed_at": self.indexed_at,
            "index_report": self.index_report,
            "chunk_count": self.chunk_count,
            "indexed_count": self.indexed_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentProcessing:
        return cls(
            document_id=int(data["document_id"]),
            name=str(data["name"]),
            media_type=MediaType(data["media_type"]),
            status=ProcessingStatus(data["status"]),
            error=str(data["error"]),
            page_count=None if data["page_count"] is None else int(data["page_count"]),
            chunking_version=str(data["chunking_version"]),
            processed_at=str(data["processed_at"]),
            report=dict(data["report"]),
            index_status=IndexStatus(data["index_status"]),
            index_model=str(data["index_model"]),
            index_error=str(data["index_error"]),
            indexed_at=str(data["indexed_at"]),
            index_report=dict(data["index_report"]),
            chunk_count=int(data["chunk_count"]),
            indexed_count=int(data["indexed_count"]),
        )
