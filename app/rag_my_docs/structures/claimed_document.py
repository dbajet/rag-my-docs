from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.constants.media_type import MediaType


class ClaimedDocument(NamedTuple):
    """A document this process has taken for processing: `run` is the number of
    this attempt, and only the run that still holds the document may store its
    results, so a reprocess asked for meanwhile is never overwritten."""

    document_id: int
    run: int
    media_type: MediaType
    content: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "run": self.run,
            "media_type": self.media_type.value,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClaimedDocument:
        return cls(
            document_id=int(data["document_id"]),
            run=int(data["run"]),
            media_type=MediaType(data["media_type"]),
            content=bytes(data["content"]),
        )
