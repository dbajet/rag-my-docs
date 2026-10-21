from __future__ import annotations

from typing import Any

import pymupdf
import pytest

from tests.support.pdf_factory import PdfFactory


def helper_opened(data: bytes) -> Any:
    """The built bytes as PyMuPDF reads them back, independently of PdfBox."""
    library: Any = pymupdf
    return library.open(stream=data, filetype="pdf")


def helper_texts(data: bytes) -> list[list[str]]:
    with helper_opened(data) as document:
        return [[block[4] for block in page.get_text("blocks")] for page in document]


def test_class() -> None:
    tested = PdfFactory
    result = (tested.WIDTH, tested.HEIGHT, tested.MARGIN, tested.GAP, tested.FIXED_DATE)
    expected = (595.0, 842.0, 72.0, 8.0, "D:20260101000000Z")
    assert result == expected


def test___init__() -> None:
    tested = PdfFactory()
    result = (tested._pymupdf is pymupdf, tested._document.page_count, tested._page, tested._cursor)
    expected = (True, 0, None, 72.0)
    assert result == expected


def test_page() -> None:
    tested = PdfFactory()
    tested._cursor = 300.0
    result = tested.page()
    assert result is tested
    exp_state = (1, 595.0, 842.0, 72.0)
    assert (tested._document.page_count, tested._page.rect.width, tested._page.rect.height, tested._cursor) == exp_state


def test_paragraph() -> None:
    tested = PdfFactory().page()
    result = tested.paragraph("First block.", 10, False).paragraph("Second block.", 14, True)
    assert result is tested
    expected = [["First block.\n", "Second block.\n"]]
    assert helper_texts(tested.built()) == expected
    with helper_opened(tested.built()) as document:
        fonts = [
            (span["font"], span["size"])
            for block in document[0].get_text("dict")["blocks"]
            for line in block["lines"]
            for span in line["spans"]
        ]
    exp_fonts = [("Helvetica", 10.0), ("Helvetica-Bold", 14.0)]
    assert fonts == exp_fonts


def test_paragraph__too_long_for_the_page() -> None:
    tested = PdfFactory().page()
    with pytest.raises(ValueError) as exc_info:
        tested.paragraph("Word " * 5000, 10, False)
    assert str(exc_info.value) == "the paragraph does not fit on the page"


def test_text_in() -> None:
    tested = PdfFactory().page()
    result = tested.text_in((300, 100, 500, 200), "Boxed text", 12, True)
    assert result is tested
    with helper_opened(tested.built()) as document:
        found = [(round(block[0]), round(block[1]), block[4]) for block in document[0].get_text("blocks")]
    expected = [(300, 100, "Boxed text\n")]
    assert found == expected
    exp_cursor = 72.0
    assert tested._cursor == exp_cursor


def test_text_in__too_long_for_its_box() -> None:
    tested = PdfFactory().page()
    with pytest.raises(ValueError) as exc_info:
        tested.text_in((300, 100, 320, 110), "Far too much text for a tiny box", 12, False)
    assert str(exc_info.value) == "the text does not fit in its box"


def test_table() -> None:
    tested = PdfFactory().page()
    result = tested.table([["Name", "Qty"], ["Apple", "3"]], 300)
    assert result is tested
    exp_cursor = 348.0
    assert tested._cursor == exp_cursor
    with helper_opened(tested.built()) as document:
        found = document[0].find_tables().tables
        tables = [(table.header.names, table.extract()) for table in found]
        lines = len(document[0].get_drawings())
    expected = [(["Name", "Qty"], [["Name", "Qty"], ["Apple", "3"]])]
    assert tables == expected
    exp_lines = 6
    assert lines == exp_lines


def test_image() -> None:
    tested = PdfFactory().page()
    result = tested.image()
    assert result is tested
    with helper_opened(tested.built()) as document:
        found = (len(document[0].get_images()), document[0].get_text().strip())
    expected = (1, "")
    assert found == expected


def test_bookmarks() -> None:
    tested = PdfFactory().page().page()
    result = tested.bookmarks([(1, "One", 1), (2, "One point one", 2)])
    assert result is tested
    with helper_opened(tested.built()) as document:
        toc = document.get_toc()
    expected = [[1, "One", 1], [2, "One point one", 2]]
    assert toc == expected


def test_built() -> None:
    tested = PdfFactory().page().paragraph("Same every time.", 10, False)
    result = tested.built()
    with helper_opened(result) as document:
        dates = (document.metadata["creationDate"], document.metadata["modDate"])
    exp_dates = ("D:20260101000000Z", "D:20260101000000Z")
    assert dates == exp_dates
    expected = PdfFactory().page().paragraph("Same every time.", 10, False).built()
    assert result == expected


@pytest.mark.parametrize(
    "bold, expected",
    [
        pytest.param(False, "helv", id="regular"),
        pytest.param(True, "hebo", id="bold"),
    ],
)
def test__font(bold: bool, expected: str) -> None:
    tested = PdfFactory()
    result = tested._font(bold)
    assert result == expected
