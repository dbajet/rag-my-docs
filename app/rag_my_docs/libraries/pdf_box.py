from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pymupdf

from rag_my_docs.constants.processing_constants import ProcessingConstants
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.outline_entry import OutlineEntry
from rag_my_docs.structures.pdf_document import PdfDocument
from rag_my_docs.structures.pdf_page import PdfPage
from rag_my_docs.structures.raw_block import RawBlock
from rag_my_docs.structures.raw_table import RawTable


class PdfBox:
    """PyMuPDF behind one door: everything read from a PDF comes out as plain
    structures (blocks with their lines, positions and fonts; tables; bookmarks;
    image counts) and nothing else in the application imports PyMuPDF.

    Nothing is interpreted here: no heading, no header, no reading order —
    those are the next stages' decisions, kept testable without a PDF. Only
    extraction artifacts are normalised: runs of spaces inside a line become
    one, and PyMuPDF joins words hyphenated across a line break."""

    # PyMuPDF's functions carry no type hints: they are reached through this one
    # reference, typed as Any, and everything they return is checked into our types here.
    _PYMUPDF: Any = pymupdf
    _FLAGS = (pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES) | pymupdf.TEXT_DEHYPHENATE
    _BOLD_FLAG = 16
    _BOLD_NAMES = ("bold", "black", "heavy")
    _SPACES = re.compile(r"\s+")

    def __init__(self) -> None:
        # PyMuPDF prints an advertisement for its machine-learning layout add-on once.
        self._PYMUPDF.no_recommend_layout()

    def read(self, content: bytes) -> PdfDocument:
        try:
            document = self._PYMUPDF.open(stream=content, filetype="pdf")
        except (RuntimeError, ValueError) as error:
            raise AppException(422, ProcessingConstants.unreadable) from error
        with document:
            if document.needs_pass:
                raise AppException(422, ProcessingConstants.protected)
            try:
                return PdfDocument(
                    pages=tuple(self.page(document[index], index + 1) for index in range(document.page_count)),
                    outline=self.outline(document),
                )
            except RuntimeError as error:
                # MuPDF parses pages lazily: a damaged page fails only when read.
                raise AppException(422, ProcessingConstants.unreadable) from error

    def page(self, page: Any, number: int) -> PdfPage:
        content = page.get_text("dict", flags=self._FLAGS)
        blocks = [self.block(raw, number) for raw in content["blocks"] if raw["type"] == 0]
        return PdfPage(
            number=number,
            width=float(page.rect.width),
            height=float(page.rect.height),
            blocks=tuple(block for block in blocks if block is not None),
            tables=self.tables(page, number),
            image_count=len(page.get_images()),
        )

    def block(self, raw: dict[str, Any], page: int) -> RawBlock | None:
        """None for a block with no visible character."""
        spans = [span for line in raw["lines"] for span in line["spans"] if span["text"].strip()]
        lines = [self.cleaned("".join(span["text"] for span in line["spans"])) for line in raw["lines"]]
        visible = tuple(line for line in lines if line)
        if not visible:
            return None
        sizes: Counter[float] = Counter()
        bold = 0
        for span in spans:
            count = len(span["text"].strip())
            sizes[round(float(span["size"]), 1)] += count
            bold += count if self.is_bold(span) else 0
        x0, y0, x1, y1 = (float(value) for value in raw["bbox"])
        return RawBlock(
            page=page,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            lines=visible,
            font_size=sizes.most_common(1)[0][0],
            bold_share=bold / sum(sizes.values()),
        )

    def tables(self, page: Any, number: int) -> tuple[RawTable, ...]:
        """The tables PyMuPDF finds from ruling lines. Its header is the first
        row unless it found the header outside the table's box."""
        result: list[RawTable] = []
        for table in page.find_tables().tables:
            rows = [tuple(self.cleaned(cell or "") for cell in row) for row in table.extract()]
            header = tuple(self.cleaned(name or "") for name in table.header.names)
            body = rows if table.header.external else rows[1:]
            x0, y0, x1, y1 = (float(value) for value in table.bbox)
            result.append(RawTable(page=number, x0=x0, y0=y0, x1=x1, y1=y1, header=header, rows=tuple(body)))
        return tuple(result)

    @classmethod
    def outline(cls, document: Any) -> tuple[OutlineEntry, ...]:
        """The bookmarks that point at a page; a bookmark to nowhere is dropped."""
        return tuple(
            OutlineEntry(level=int(level), title=cls.cleaned(str(title)), page=int(page))
            for level, title, page in document.get_toc(simple=True)
            if int(page) >= 1 and cls.cleaned(str(title))
        )

    @classmethod
    def is_bold(cls, span: dict[str, Any]) -> bool:
        font = str(span["font"]).lower()
        return bool(int(span["flags"]) & cls._BOLD_FLAG) or any(name in font for name in cls._BOLD_NAMES)

    @classmethod
    def cleaned(cls, text: str) -> str:
        return cls._SPACES.sub(" ", text).strip()
