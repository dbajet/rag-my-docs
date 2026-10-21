from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.constants.block_kind import BlockKind


class TextBlock(NamedTuple):
    """A unit of extracted text in reading order: a paragraph, a heading, a
    table, or a page header or footer. Its text is the source every chunk and
    section title points back into, by block id and character offsets."""

    block_id: int
    page: int
    kind: BlockKind
    text: str
    font_size: float
    bold_share: float
    line_count: int
    # Where it sits, as shares of the page height: page furniture lives at the edges.
    top: float
    bottom: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "page": self.page,
            "kind": self.kind.value,
            "text": self.text,
            "font_size": self.font_size,
            "bold_share": self.bold_share,
            "line_count": self.line_count,
            "top": self.top,
            "bottom": self.bottom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextBlock:
        return cls(
            block_id=int(data["block_id"]),
            page=int(data["page"]),
            kind=BlockKind(data["kind"]),
            text=str(data["text"]),
            font_size=float(data["font_size"]),
            bold_share=float(data["bold_share"]),
            line_count=int(data["line_count"]),
            top=float(data["top"]),
            bottom=float(data["bottom"]),
        )
