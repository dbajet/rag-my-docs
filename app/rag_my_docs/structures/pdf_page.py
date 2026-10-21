from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.structures.raw_block import RawBlock
from rag_my_docs.structures.raw_table import RawTable


class PdfPage(NamedTuple):
    """One page as read, before any interpretation. Its images are only counted:
    a page with images and no text is probably a scan."""

    number: int
    width: float
    height: float
    blocks: tuple[RawBlock, ...]
    tables: tuple[RawTable, ...]
    image_count: int

    def char_count(self) -> int:
        return sum(len(line.strip()) for block in self.blocks for line in block.lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "width": self.width,
            "height": self.height,
            "blocks": [block.to_dict() for block in self.blocks],
            "tables": [table.to_dict() for table in self.tables],
            "image_count": self.image_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PdfPage:
        return cls(
            number=int(data["number"]),
            width=float(data["width"]),
            height=float(data["height"]),
            blocks=tuple(RawBlock.from_dict(block) for block in data["blocks"]),
            tables=tuple(RawTable.from_dict(table) for table in data["tables"]),
            image_count=int(data["image_count"]),
        )
