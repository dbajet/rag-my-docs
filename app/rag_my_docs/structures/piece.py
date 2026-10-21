from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.structures.source_span import SourceSpan


class Piece(NamedTuple):
    """What chunks are made of: a sentence, a fragment of an over-long one, or
    a table row. `lead` is the text that separated it from the previous piece
    of the same paragraph ("" for a paragraph's first): chunks rebuild the
    original spacing from it."""

    text: str
    token_count: int
    spans: tuple[SourceSpan, ...]
    start_page: int
    end_page: int
    paragraph: int
    lead: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_count": self.token_count,
            "spans": [span.to_dict() for span in self.spans],
            "start_page": self.start_page,
            "end_page": self.end_page,
            "paragraph": self.paragraph,
            "lead": self.lead,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Piece:
        return cls(
            text=str(data["text"]),
            token_count=int(data["token_count"]),
            spans=tuple(SourceSpan.from_dict(span) for span in data["spans"]),
            start_page=int(data["start_page"]),
            end_page=int(data["end_page"]),
            paragraph=int(data["paragraph"]),
            lead=str(data["lead"]),
        )
