from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.structures.chunk import Chunk
from rag_my_docs.structures.section import Section
from rag_my_docs.structures.text_block import TextBlock


class ProcessingResult(NamedTuple):
    """What the pipeline made of one PDF. `report` holds counts and page numbers
    only, never document text: it is stored and shown unsealed."""

    page_count: int
    blocks: tuple[TextBlock, ...]
    sections: tuple[Section, ...]
    chunks: tuple[Chunk, ...]
    report: dict[str, Any]

    def has_text(self) -> bool:
        return bool(self.chunks) or any(section.heading_block_id is not None for section in self.sections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_count": self.page_count,
            "blocks": [block.to_dict() for block in self.blocks],
            "sections": [section.to_dict() for section in self.sections],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "report": self.report,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessingResult:
        return cls(
            page_count=int(data["page_count"]),
            blocks=tuple(TextBlock.from_dict(block) for block in data["blocks"]),
            sections=tuple(Section.from_dict(section) for section in data["sections"]),
            chunks=tuple(Chunk.from_dict(chunk) for chunk in data["chunks"]),
            report=dict(data["report"]),
        )
