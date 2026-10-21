from __future__ import annotations

from typing import Any, NamedTuple


class ParagraphSegment(NamedTuple):
    """One block's share of a paragraph: the paragraph's characters from
    `offset` on are the block's from character 0 on, for `length` characters."""

    block_id: int
    page: int
    offset: int
    length: int

    def to_dict(self) -> dict[str, Any]:
        return {"block_id": self.block_id, "page": self.page, "offset": self.offset, "length": self.length}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParagraphSegment:
        return cls(
            block_id=int(data["block_id"]),
            page=int(data["page"]),
            offset=int(data["offset"]),
            length=int(data["length"]),
        )
