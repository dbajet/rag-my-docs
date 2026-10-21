from __future__ import annotations

from rag_my_docs.structures.outline_entry import OutlineEntry
from rag_my_docs.structures.pdf_document import PdfDocument
from rag_my_docs.structures.pdf_page import PdfPage
from tests.conftest import is_namedtuple


def helper_instance() -> PdfDocument:
    page = PdfPage(number=1, width=595.0, height=842.0, blocks=(), tables=(), image_count=1)
    return PdfDocument(pages=(page,), outline=(OutlineEntry(level=1, title="Introduction", page=1),))


def test_class() -> None:
    tested = PdfDocument
    fields = {"pages": "tuple[PdfPage, ...]", "outline": "tuple[OutlineEntry, ...]"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = helper_instance()
    result = tested.to_dict()
    expected = {
        "pages": [{"number": 1, "width": 595.0, "height": 842.0, "blocks": [], "tables": [], "image_count": 1}],
        "outline": [{"level": 1, "title": "Introduction", "page": 1}],
    }
    assert result == expected


def test_from_dict() -> None:
    tested = PdfDocument
    result = tested.from_dict(
        {
            "pages": [{"number": 1, "width": 595, "height": 842, "blocks": [], "tables": [], "image_count": 1}],
            "outline": [{"level": "1", "title": "Introduction", "page": "1"}],
        }
    )
    expected = helper_instance()
    assert result == expected
