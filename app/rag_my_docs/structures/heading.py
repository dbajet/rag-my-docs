from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.constants.section_source import SectionSource


class Heading(NamedTuple):
    """Where a section starts. `block_id` is the heading's own block, or None
    for a bookmark whose title was not found in the text: that section then
    starts at `start_block_id`, the first block of the page it points to."""

    block_id: int | None
    start_block_id: int
    level: int
    number: str
    title: str
    source: SectionSource

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "start_block_id": self.start_block_id,
            "level": self.level,
            "number": self.number,
            "title": self.title,
            "source": self.source.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Heading:
        return cls(
            block_id=None if data["block_id"] is None else int(data["block_id"]),
            start_block_id=int(data["start_block_id"]),
            level=int(data["level"]),
            number=str(data["number"]),
            title=str(data["title"]),
            source=SectionSource(data["source"]),
        )
