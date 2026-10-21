from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.constants.section_source import SectionSource


class StoredSection(NamedTuple):
    """A section as stored, with its database ids and how many chunks it holds."""

    section_id: int
    parent_id: int | None
    position: int
    level: int
    number: str
    title: str
    source: SectionSource
    start_page: int
    end_page: int
    heading_block_id: int | None
    chunk_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "parent_id": self.parent_id,
            "position": self.position,
            "level": self.level,
            "number": self.number,
            "title": self.title,
            "source": self.source.value,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "heading_block_id": self.heading_block_id,
            "chunk_count": self.chunk_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoredSection:
        return cls(
            section_id=int(data["section_id"]),
            parent_id=None if data["parent_id"] is None else int(data["parent_id"]),
            position=int(data["position"]),
            level=int(data["level"]),
            number=str(data["number"]),
            title=str(data["title"]),
            source=SectionSource(data["source"]),
            start_page=int(data["start_page"]),
            end_page=int(data["end_page"]),
            heading_block_id=None if data["heading_block_id"] is None else int(data["heading_block_id"]),
            chunk_count=int(data["chunk_count"]),
        )
