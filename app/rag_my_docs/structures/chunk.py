from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.structures.source_span import SourceSpan


class Chunk(NamedTuple):
    """A retrieval unit. Its text is original document text: a context prefix
    (the overlap sentence, or a table's header row), then its own content.
    `spans` cover its own content only, so the chunks' spans, in order, never
    overlap and together rebuild the document's body text."""

    position: int
    section_position: int
    kind: ChunkKind
    text: str
    prefix_chars: int
    token_count: int
    start_page: int
    end_page: int
    spans: tuple[SourceSpan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "section_position": self.section_position,
            "kind": self.kind.value,
            "text": self.text,
            "prefix_chars": self.prefix_chars,
            "token_count": self.token_count,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "spans": [span.to_dict() for span in self.spans],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            position=int(data["position"]),
            section_position=int(data["section_position"]),
            kind=ChunkKind(data["kind"]),
            text=str(data["text"]),
            prefix_chars=int(data["prefix_chars"]),
            token_count=int(data["token_count"]),
            start_page=int(data["start_page"]),
            end_page=int(data["end_page"]),
            spans=tuple(SourceSpan.from_dict(span) for span in data["spans"]),
        )
