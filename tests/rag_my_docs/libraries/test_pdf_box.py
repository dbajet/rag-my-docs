from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, call, patch

import pymupdf
import pytest

from rag_my_docs.libraries.pdf_box import PdfBox
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.outline_entry import OutlineEntry
from rag_my_docs.structures.raw_block import RawBlock
from rag_my_docs.structures.raw_table import RawTable
from tests.support.pdf_factory import PdfFactory


def helper_shape(block: RawBlock) -> tuple[Any, ...]:
    """A block with its coordinates rounded: PyMuPDF measures in fractions of a point."""
    return (
        block.page,
        round(block.x0),
        round(block.y0),
        round(block.x1),
        round(block.y1),
        block.lines,
        block.font_size,
        block.bold_share,
    )


def helper_two_pages() -> bytes:
    return (
        PdfFactory()
        .page()
        .paragraph("1. Introduction", 14, True)
        .paragraph("The   first   page   says   hello.", 10, False)
        .table([["Name", "Qty"], ["Apple", "3"]], 300)
        .page()
        .paragraph("The second page.", 10, False)
        .bookmarks([(1, "1. Introduction", 1), (2, "Nowhere", -1)])
        .built()
    )


def helper_opened(data: bytes) -> Any:
    library: Any = pymupdf
    return library.open(stream=data, filetype="pdf")


def helper_span(text: str, size: float, flags: int, font: str) -> dict[str, Any]:
    return {"text": text, "size": size, "flags": flags, "font": font}


def test_class() -> None:
    tested = PdfBox
    result = (tested._PYMUPDF is pymupdf, tested._BOLD_FLAG, tested._BOLD_NAMES, tested._SPACES.pattern)
    expected = (True, 16, ("bold", "black", "heavy"), r"\s+")
    assert result == expected
    exp_flags = (pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES) | pymupdf.TEXT_DEHYPHENATE
    assert tested._FLAGS == exp_flags


def test___init__() -> None:
    with patch.object(pymupdf, "no_recommend_layout", side_effect=[None]) as mock_no_recommend:
        result = PdfBox()
    assert isinstance(result, PdfBox) is True
    exp_calls = [call()]
    assert mock_no_recommend.mock_calls == exp_calls


def test_read() -> None:
    tested = PdfBox()
    result = tested.read(helper_two_pages())
    pages = [
        (page.number, page.width, page.height, page.image_count, [helper_shape(block) for block in page.blocks])
        for page in result.pages
    ]
    expected = [
        (
            1,
            595.0,
            842.0,
            0,
            [
                (1, 72, 72, 169, 91, ("1. Introduction",), 14.0, 1.0),
                (1, 72, 104, 205, 117, ("The first page says hello.",), 10.0, 0.0),
                (1, 76, 303, 317, 317, ("Name", "Qty"), 10.0, 0.0),
                (1, 76, 323, 307, 337, ("Apple", "3"), 10.0, 0.0),
            ],
        ),
        (2, 595.0, 842.0, 0, [(2, 72, 72, 152, 86, ("The second page.",), 10.0, 0.0)]),
    ]
    assert pages == expected
    exp_tables = [
        (RawTable(page=1, x0=72.0, y0=300.0, x1=523.0, y1=340.0, header=("Name", "Qty"), rows=(("Apple", "3"),)),),
        (),
    ]
    assert [page.tables for page in result.pages] == exp_tables
    exp_outline = (OutlineEntry(level=1, title="1. Introduction", page=1),)
    assert result.outline == exp_outline


def test_read__blank_and_scanned_pages() -> None:
    tested = PdfBox()
    result = tested.read(PdfFactory().page().page().image().built())
    pages = [(page.number, page.blocks, page.tables, page.image_count, page.char_count()) for page in result.pages]
    expected = [(1, (), (), 0, 0), (2, (), (), 1, 0)]
    assert pages == expected


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"%PDF-1.4 garbage", id="truncated"),
        pytest.param(b"not a pdf at all", id="not-a-pdf"),
    ],
)
def test_read__malformed(content: bytes) -> None:
    tested = PdfBox()
    with pytest.raises(AppException) as exc_info:
        tested.read(content)
    expected = (422, "This PDF could not be read: the file is damaged or is not a PDF.")
    assert (exc_info.value.status_code, exc_info.value.message) == expected


def test_read__password_protected() -> None:
    tested = PdfBox()
    with helper_opened(PdfFactory().page().paragraph("Secret.", 10, False).built()) as document:
        protected = document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="y")
    with pytest.raises(AppException) as exc_info:
        tested.read(protected)
    expected = (422, "This PDF is protected by a password.")
    assert (exc_info.value.status_code, exc_info.value.message) == expected


def test_read__a_page_failing_when_read() -> None:
    tested = PdfBox()
    with (
        patch.object(PdfBox, "page", side_effect=[RuntimeError("broken page")]) as mock_page,
        pytest.raises(AppException) as exc_info,
    ):
        tested.read(PdfFactory().page().built())
    expected = (422, "This PDF could not be read: the file is damaged or is not a PDF.")
    assert (exc_info.value.status_code, exc_info.value.message) == expected
    # The first argument is PyMuPDF's own page object; the page number is ours.
    exp_calls = [call(ANY, 1)]
    assert mock_page.mock_calls == exp_calls


def test_page() -> None:
    tested = PdfBox()
    data = PdfFactory().page().paragraph("Only text.", 10, False).image().built()
    with helper_opened(data) as document:
        result = tested.page(document[0], 7)
    expected = (7, 595.0, 842.0, [(7, 72, 72, 114, 86, ("Only text.",), 10.0, 0.0)], (), 1)
    found = (
        result.number,
        result.width,
        result.height,
        [helper_shape(block) for block in result.blocks],
        result.tables,
        result.image_count,
    )
    assert found == expected


def test_block() -> None:
    tested = PdfBox()
    raw = {
        "bbox": (10, 20, 110, 60),
        "lines": [
            {"spans": [helper_span("Bold  ", 12.04, 16, "Helvetica-Bold"), helper_span("words", 12.0, 0, "Helvetica")]},
            {"spans": [helper_span("   ", 9.0, 0, "Helvetica")]},
            {"spans": [helper_span(" second\tline ", 12.0, 0, "Helvetica")]},
        ],
    }
    result = tested.block(raw, 4)
    expected = RawBlock(
        page=4,
        x0=10.0,
        y0=20.0,
        x1=110.0,
        y1=60.0,
        lines=("Bold words", "second line"),
        font_size=12.0,
        # "Bold" is 4 of the 20 visible characters (blank spans count for nothing).
        bold_share=0.2,
    )
    assert result == expected


def test_block__nothing_visible() -> None:
    tested = PdfBox()
    raw = {"bbox": (0, 0, 1, 1), "lines": [{"spans": [helper_span("  ", 10.0, 0, "Helvetica")]}]}
    result = tested.block(raw, 1)
    assert result is None


def test_tables() -> None:
    tested = PdfBox()
    data = PdfFactory().page().table([["Name", "Notes"], ["Apple", "red   and   round"], ["Pear", ""]], 300).built()
    with helper_opened(data) as document:
        result = tested.tables(document[0], 2)
    expected = (
        RawTable(
            page=2,
            x0=72.0,
            y0=300.0,
            x1=523.0,
            y1=360.0,
            header=("Name", "Notes"),
            rows=(("Apple", "red and round"), ("Pear", "")),
        ),
    )
    assert result == expected


def test_tables__header_outside_the_table() -> None:
    tested = PdfBox()
    table = SimpleNamespace(
        extract=lambda: [["Apple", None], ["Pear", "2"]],
        header=SimpleNamespace(names=["Name", None], external=True),
        bbox=(1, 2, 3, 4),
    )
    page = SimpleNamespace(find_tables=lambda: SimpleNamespace(tables=[table]))
    result = tested.tables(page, 5)
    expected = (
        RawTable(page=5, x0=1.0, y0=2.0, x1=3.0, y1=4.0, header=("Name", ""), rows=(("Apple", ""), ("Pear", "2"))),
    )
    assert result == expected


def test_outline() -> None:
    tested = PdfBox
    document = SimpleNamespace(
        get_toc=lambda simple: [[1, " A   title ", 2], [2, "Gone", -1], [1, "   ", 3], [2, "Sub", 3]] if simple else []
    )
    result = tested.outline(document)
    expected = (OutlineEntry(level=1, title="A title", page=2), OutlineEntry(level=2, title="Sub", page=3))
    assert result == expected


@pytest.mark.parametrize(
    "span, expected",
    [
        pytest.param(helper_span("x", 10.0, 16, "Helvetica"), True, id="bold-flag"),
        pytest.param(helper_span("x", 10.0, 20, "Helvetica"), True, id="bold-among-flags"),
        pytest.param(helper_span("x", 10.0, 0, "Arial-BoldMT"), True, id="bold-name"),
        pytest.param(helper_span("x", 10.0, 0, "Inter-Black"), True, id="black-name"),
        pytest.param(helper_span("x", 10.0, 0, "Font-Heavy"), True, id="heavy-name"),
        pytest.param(helper_span("x", 10.0, 4, "Helvetica"), False, id="regular"),
    ],
)
def test_is_bold(span: dict[str, Any], expected: bool) -> None:
    tested = PdfBox
    result = tested.is_bold(span)
    assert result is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("  two   spaces\tand\na line  ", "two spaces and a line", id="whitespace"),
        pytest.param("Dr. Smith paid $3.50.", "Dr. Smith paid $3.50.", id="punctuation-kept"),
        pytest.param(" \n\t ", "", id="blank"),
    ],
)
def test_cleaned(text: str, expected: str) -> None:
    tested = PdfBox
    result = tested.cleaned(text)
    assert result == expected
