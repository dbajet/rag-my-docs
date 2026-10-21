from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus


class Document(NamedTuple):
    """A stored document as a list shows it: its name opened, its content left
    sealed in the database until someone downloads it."""

    document_id: int
    user_id: int
    name: str
    media_type: MediaType
    size_bytes: int
    created_at: str
    processing_status: ProcessingStatus
    index_status: IndexStatus
    chunk_count: int
    # Chunks with an embedding from the current or an earlier model.
    indexed_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "user_id": self.user_id,
            "name": self.name,
            "media_type": self.media_type.value,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "processing_status": self.processing_status.value,
            "index_status": self.index_status.value,
            "chunk_count": self.chunk_count,
            "indexed_count": self.indexed_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        return cls(
            document_id=int(data["document_id"]),
            user_id=int(data["user_id"]),
            name=str(data["name"]),
            media_type=MediaType(data["media_type"]),
            size_bytes=int(data["size_bytes"]),
            created_at=str(data["created_at"]),
            processing_status=ProcessingStatus(data["processing_status"]),
            index_status=IndexStatus(data["index_status"]),
            chunk_count=int(data["chunk_count"]),
            indexed_count=int(data["indexed_count"]),
        )
