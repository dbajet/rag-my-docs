from __future__ import annotations

import re

from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.constants.processing_constants import ProcessingConstants
from rag_my_docs.structures.pdf_document import PdfDocument
from rag_my_docs.structures.pdf_page import PdfPage
from rag_my_docs.structures.raw_block import RawBlock
from rag_my_docs.structures.raw_table import RawTable
from rag_my_docs.structures.text_block import TextBlock


class PageLayout:
    """Puts each page's blocks and tables in reading order and turns them into
    numbered text blocks: the one sequence every later stage points into.

    Reading order, explained: blocks are taken top to bottom. A block that
    crosses the middle of the page (a title, a full-width paragraph, a table)
    ends a "band"; inside a band, when there really are two columns — at least
    two blocks on each side of the middle, side by side — the left column is
    read before the right one. Otherwise the band is read top to bottom.

    A block lying inside a table is the table's own text, already kept as
    rows: it is dropped here so no table text is read twice."""

    # A line starting like a list item keeps its line break, so items stay items.
    _LIST_ITEM = re.compile(r"^(?:[•◦▪‣⁃·*\-–]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s")

    def blocks(self, document: PdfDocument) -> list[TextBlock]:
        result: list[TextBlock] = []
        for page in document.pages:
            for item in self.ordered(page):
                result.append(self.text_block(len(result), page, item))
        return result

    def ordered(self, page: PdfPage) -> list[RawBlock | RawTable]:
        middle = page.width / 2
        tolerance = page.width * ProcessingConstants.column_tolerance
        items: list[RawBlock | RawTable] = [block for block in page.blocks if not self.is_in_table(block, page.tables)]
        items.extend(page.tables)
        result: list[RawBlock | RawTable] = []
        band: list[RawBlock | RawTable] = []
        for item in sorted(items, key=lambda candidate: (candidate.y0, candidate.x0)):
            if item.x0 < middle - tolerance and item.x1 > middle + tolerance:
                result.extend(self.band_order(band, middle, tolerance))
                band = []
                result.append(item)
            else:
                band.append(item)
        result.extend(self.band_order(band, middle, tolerance))
        return result

    @classmethod
    def band_order(
        cls,
        band: list[RawBlock | RawTable],
        middle: float,
        tolerance: float,
    ) -> list[RawBlock | RawTable]:
        left = [item for item in band if item.x1 <= middle + tolerance]
        right = [item for item in band if item.x1 > middle + tolerance]
        side_by_side = any(a.y0 < b.y1 and b.y0 < a.y1 for a in left for b in right)
        if len(left) >= 2 and len(right) >= 2 and side_by_side:
            return left + right
        return sorted(band, key=lambda item: (item.y0, item.x0))

    @classmethod
    def is_in_table(cls, block: RawBlock, tables: tuple[RawTable, ...]) -> bool:
        area = max((block.x1 - block.x0) * (block.y1 - block.y0), 1.0)
        for table in tables:
            width = min(block.x1, table.x1) - max(block.x0, table.x0)
            height = min(block.y1, table.y1) - max(block.y0, table.y0)
            if width > 0 and height > 0 and width * height / area >= ProcessingConstants.table_overlap:
                return True
        return False

    def text_block(self, block_id: int, page: PdfPage, item: RawBlock | RawTable) -> TextBlock:
        if isinstance(item, RawTable):
            return TextBlock(
                block_id=block_id,
                page=page.number,
                kind=BlockKind.TABLE,
                text=self.table_text(item),
                font_size=0.0,
                bold_share=0.0,
                line_count=len(item.rows) + 1,
                top=item.y0 / page.height,
                bottom=item.y1 / page.height,
            )
        return TextBlock(
            block_id=block_id,
            page=page.number,
            kind=BlockKind.PARAGRAPH,
            text=self.joined(item.lines),
            font_size=item.font_size,
            bold_share=item.bold_share,
            line_count=len(item.lines),
            top=item.y0 / page.height,
            bottom=item.y1 / page.height,
        )

    @classmethod
    def joined(cls, lines: tuple[str, ...]) -> str:
        """A block's lines as one paragraph: wrapped lines join with a space,
        a list item starts on its own line."""
        result = lines[0]
        for line in lines[1:]:
            result += ("\n" if cls._LIST_ITEM.match(line) else " ") + line
        return result

    @classmethod
    def table_text(cls, table: RawTable) -> str:
        """The header row, then one line per row: `| cell | cell |`. A row is
        one line, so a chunk can take whole rows and never half of one."""
        return "\n".join(cls.row_text(row) for row in (table.header, *table.rows))

    @classmethod
    def row_text(cls, cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(cells) + " |"
