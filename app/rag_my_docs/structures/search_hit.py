from __future__ import annotations

from typing import Any, NamedTuple


class SearchHit(NamedTuple):
    """One retrieved chunk: where it comes from, its original text, and how
    close it is to the question (cosine similarity, 1 is the same direction)."""

    rank: int
    chunk_id: int
    chunk_position: int
    document_id: int
    document_name: str
    section_id: int
    section_path: str
    start_page: int
    end_page: int
    text: str
    prefix_chars: int
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "chunk_position": self.chunk_position,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "section_id": self.section_id,
            "section_path": self.section_path,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "text": self.text,
            "prefix_chars": self.prefix_chars,
            "similarity": self.similarity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchHit:
        return cls(
            rank=int(data["rank"]),
            chunk_id=int(data["chunk_id"]),
            chunk_position=int(data["chunk_position"]),
            document_id=int(data["document_id"]),
            document_name=str(data["document_name"]),
            section_id=int(data["section_id"]),
            section_path=str(data["section_path"]),
            start_page=int(data["start_page"]),
            end_page=int(data["end_page"]),
            text=str(data["text"]),
            prefix_chars=int(data["prefix_chars"]),
            similarity=float(data["similarity"]),
        )
