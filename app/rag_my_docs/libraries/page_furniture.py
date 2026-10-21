from __future__ import annotations

import math
import re

from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.constants.processing_constants import ProcessingConstants
from rag_my_docs.structures.text_block import TextBlock


class PageFurniture:
    """Finds running headers and footers: short blocks at the top or bottom
    edge of the page whose text repeats on many pages once digits are ignored
    ("Page 3 of 10" and "Page 4 of 10" are the same line), or that carry the
    printed page number at either end. They are marked, not deleted: the
    inspection view shows them, and chunks leave them out.

    A line at the edge of one page only is ordinary text; a document of one
    page has no furniture."""

    _DIGITS = re.compile(r"\d+")
    _SPACES = re.compile(r"\s+")
    _LEADING_NUMBER = re.compile(r"^(\d+)\b")
    _TRAILING_NUMBER = re.compile(r"\b(\d+)$")

    def marked(self, blocks: list[TextBlock]) -> list[TextBlock]:
        pages = {block.page for block in blocks}
        needed = max(
            ProcessingConstants.furniture_min_pages,
            math.ceil(ProcessingConstants.furniture_page_share * len(pages)),
        )
        seen: dict[str, set[int]] = {}
        for block in blocks:
            if self.is_candidate(block):
                seen.setdefault(self.key(block.text), set()).add(block.page)
        repeated = {key for key, on_pages in seen.items() if len(on_pages) >= needed}
        numbered = self.numbered(blocks, needed)
        return [
            self.as_furniture(block)
            if self.is_candidate(block) and (self.key(block.text) in repeated or block.block_id in numbered)
            else block
            for block in blocks
        ]

    @classmethod
    def numbered(cls, blocks: list[TextBlock], needed: int) -> set[int]:
        """The ids of the lines that carry the printed page number: a number at
        either end of the line that follows the page with one fixed offset (a
        cover may come before page 1) on at least `needed` pages. The text around
        it does not matter — a printed book puts the number on the left of even
        pages and on the right of odd ones, beside each chapter's own title, so
        no two of those lines need to repeat."""
        pages_by_offset: dict[int, set[int]] = {}
        for block in blocks:
            if cls.is_candidate(block):
                for number in cls.edge_numbers(block.text):
                    pages_by_offset.setdefault(number - block.page, set()).add(block.page)
        offsets = {offset for offset, pages in pages_by_offset.items() if len(pages) >= needed}
        return {
            block.block_id
            for block in blocks
            if cls.is_candidate(block)
            and any(number - block.page in offsets for number in cls.edge_numbers(block.text))
        }

    @classmethod
    def edge_numbers(cls, text: str) -> set[int]:
        """The numbers that start or end the line: where a page number sits."""
        stripped = text.strip()
        found = (cls._LEADING_NUMBER.search(stripped), cls._TRAILING_NUMBER.search(stripped))
        return {int(match.group(1)) for match in found if match is not None}

    @classmethod
    def is_candidate(cls, block: TextBlock) -> bool:
        margin = ProcessingConstants.furniture_margin
        is_at_edge = block.bottom <= margin or block.top >= 1 - margin
        return block.kind is BlockKind.PARAGRAPH and is_at_edge and block.line_count <= 2

    @classmethod
    def key(cls, text: str) -> str:
        return cls._SPACES.sub(" ", cls._DIGITS.sub("#", text)).strip().casefold()

    @classmethod
    def as_furniture(cls, block: TextBlock) -> TextBlock:
        return block._replace(kind=BlockKind.HEADER if block.top < 0.5 else BlockKind.FOOTER)
