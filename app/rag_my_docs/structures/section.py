from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.constants.section_source import SectionSource


class Section(NamedTuple):
    """A logical part of the document, as detected: its place in the hierarchy
    and the blocks that belong to it directly (its subsections' are theirs).
    An untitled section has an empty title: nothing is invented."""

    position: int
    parent_position: int | None
    level: int
    number: str
    title: str
    source: SectionSource
    heading_block_id: int | None
    block_ids: tuple[int, ...]
    start_page: int
    end_page: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "parent_position": self.parent_position,
            "level": self.level,
            "number": self.number,
            "title": self.title,
            "source": self.source.value,
            "heading_block_id": self.heading_block_id,
            "block_ids": list(self.block_ids),
            "start_page": self.start_page,
            "end_page": self.end_page,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Section:
        return cls(
            position=int(data["position"]),
            parent_position=None if data["parent_position"] is None else int(data["parent_position"]),
            level=int(data["level"]),
            number=str(data["number"]),
            title=str(data["title"]),
            source=SectionSource(data["source"]),
            heading_block_id=None if data["heading_block_id"] is None else int(data["heading_block_id"]),
            block_ids=tuple(int(block_id) for block_id in data["block_ids"]),
            start_page=int(data["start_page"]),
            end_page=int(data["end_page"]),
        )
