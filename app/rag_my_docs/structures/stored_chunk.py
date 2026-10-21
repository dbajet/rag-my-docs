from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.structures.source_span import SourceSpan


class StoredChunk(NamedTuple):
    """A chunk as stored, with its database ids and the strategy that made it."""

    chunk_id: int
    section_id: int
    position: int
    kind: ChunkKind
    text: str
    prefix_chars: int
    token_count: int
    start_page: int
    end_page: int
    spans: tuple[SourceSpan, ...]
    strategy: str
    # The model its embedding came from; "" while it has none.
    embedding_model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "section_id": self.section_id,
            "position": self.position,
            "kind": self.kind.value,
            "text": self.text,
            "prefix_chars": self.prefix_chars,
            "token_count": self.token_count,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "spans": [span.to_dict() for span in self.spans],
            "strategy": self.strategy,
            "embedding_model": self.embedding_model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoredChunk:
        return cls(
            chunk_id=int(data["chunk_id"]),
            section_id=int(data["section_id"]),
            position=int(data["position"]),
            kind=ChunkKind(data["kind"]),
            text=str(data["text"]),
            prefix_chars=int(data["prefix_chars"]),
            token_count=int(data["token_count"]),
            start_page=int(data["start_page"]),
            end_page=int(data["end_page"]),
            spans=tuple(SourceSpan.from_dict(span) for span in data["spans"]),
            strategy=str(data["strategy"]),
            embedding_model=str(data["embedding_model"]),
        )
