from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.structures.paragraph_segment import ParagraphSegment
from rag_my_docs.structures.source_span import SourceSpan


class Paragraph(NamedTuple):
    """A paragraph's text, usually one block, or two when a paragraph runs on
    across a page break; the segments map its characters back to the blocks."""

    text: str
    segments: tuple[ParagraphSegment, ...]

    def spans(self, start: int, end: int) -> tuple[SourceSpan, ...]:
        """The block characters behind the paragraph's [start, end)."""
        result: list[SourceSpan] = []
        for segment in self.segments:
            low = max(start, segment.offset)
            high = min(end, segment.offset + segment.length)
            if low < high:
                result.append(
                    SourceSpan(block_id=segment.block_id, start=low - segment.offset, end=high - segment.offset)
                )
        return tuple(result)

    def pages(self, start: int, end: int) -> tuple[int, int]:
        """The first and last page of the paragraph's [start, end)."""
        touched = [
            segment.page
            for segment in self.segments
            if max(start, segment.offset) < min(end, segment.offset + segment.length)
        ]
        return min(touched), max(touched)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "segments": [segment.to_dict() for segment in self.segments]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paragraph:
        return cls(
            text=str(data["text"]),
            segments=tuple(ParagraphSegment.from_dict(segment) for segment in data["segments"]),
        )
