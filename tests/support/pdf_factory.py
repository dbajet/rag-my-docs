from __future__ import annotations

from typing import Any

import pymupdf


class PdfFactory:
    """Builds small PDFs for the tests, the same bytes on every run: text in
    the base-14 Helvetica fonts (regular and bold), ruled tables, bookmarks,
    an image standing for a scanned page. Every layout the pipeline must
    handle is described in a few lines of a test, not kept as a binary file.

    Text flows down the page from a cursor (`paragraph`), or is placed in a
    given box (`text_in`) for columns and page edges. Only the characters
    Helvetica's standard encoding has are reliable: plain ASCII in tests."""

    WIDTH = 595.0
    HEIGHT = 842.0
    MARGIN = 72.0
    GAP = 8.0
    FIXED_DATE = "D:20260101000000Z"
    _REGULAR = "helv"
    _BOLD = "hebo"

    def __init__(self) -> None:
        # PyMuPDF's functions carry no type hints; tests reach them through Any.
        self._pymupdf: Any = pymupdf
        self._document: Any = self._pymupdf.open()
        self._page: Any = None
        self._cursor = self.MARGIN

    def page(self) -> PdfFactory:
        self._page = self._document.new_page(width=self.WIDTH, height=self.HEIGHT)
        self._cursor = self.MARGIN
        return self

    def paragraph(self, text: str, size: float, bold: bool) -> PdfFactory:
        """A block of wrapped text across the page width, below the previous one."""
        bottom = self.HEIGHT - self.MARGIN
        box = self._pymupdf.Rect(self.MARGIN, self._cursor, self.WIDTH - self.MARGIN, bottom)
        spare = self._page.insert_textbox(box, text, fontsize=size, fontname=self._font(bold))
        if spare < 0:
            raise ValueError("the paragraph does not fit on the page")
        self._cursor = bottom - spare + self.GAP
        return self

    def text_in(self, box: tuple[float, float, float, float], text: str, size: float, bold: bool) -> PdfFactory:
        spare = self._page.insert_textbox(self._pymupdf.Rect(*box), text, fontsize=size, fontname=self._font(bold))
        if spare < 0:
            raise ValueError("the text does not fit in its box")
        return self

    def table(self, rows: list[list[str]], top: float) -> PdfFactory:
        """A table with a ruling line around every cell, as PyMuPDF finds them."""
        width = (self.WIDTH - 2 * self.MARGIN) / len(rows[0])
        height = 20.0
        for row_index, row in enumerate(rows):
            for column_index, cell in enumerate(row):
                point = (self.MARGIN + column_index * width + 4, top + row_index * height + 14)
                self._page.insert_text(point, cell, fontsize=10, fontname=self._REGULAR)
        for row_index in range(len(rows) + 1):
            y = top + row_index * height
            self._page.draw_line((self.MARGIN, y), (self.WIDTH - self.MARGIN, y))
        for column_index in range(len(rows[0]) + 1):
            x = self.MARGIN + column_index * width
            self._page.draw_line((x, top), (x, top + len(rows) * height))
        self._cursor = top + len(rows) * height + self.GAP
        return self

    def image(self) -> PdfFactory:
        """A small grey image covering the page, as a scanner leaves."""
        picture = self._pymupdf.Pixmap(self._pymupdf.csGRAY, self._pymupdf.IRect(0, 0, 8, 8), False)
        picture.set_rect(picture.irect, (128,))
        self._page.insert_image(self._page.rect, pixmap=picture)
        return self

    def bookmarks(self, entries: list[tuple[int, str, int]]) -> PdfFactory:
        self._document.set_toc([[level, title, page] for level, title, page in entries])
        return self

    def built(self) -> bytes:
        """Without dates or a random file id, so the same calls give the same bytes."""
        self._document.set_metadata({"creationDate": self.FIXED_DATE, "modDate": self.FIXED_DATE})
        result: bytes = self._document.tobytes(no_new_id=True)
        return result

    def _font(self, bold: bool) -> str:
        return self._BOLD if bold else self._REGULAR
