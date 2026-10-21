from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.structures.outline_entry import OutlineEntry
from rag_my_docs.structures.pdf_page import PdfPage


class PdfDocument(NamedTuple):
    """Everything read from a PDF: its pages and its bookmarks."""

    pages: tuple[PdfPage, ...]
    outline: tuple[OutlineEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": [page.to_dict() for page in self.pages],
            "outline": [entry.to_dict() for entry in self.outline],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PdfDocument:
        return cls(
            pages=tuple(PdfPage.from_dict(page) for page in data["pages"]),
            outline=tuple(OutlineEntry.from_dict(entry) for entry in data["outline"]),
        )
