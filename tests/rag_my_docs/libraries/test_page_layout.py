from __future__ import annotations

import pytest

from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.libraries.page_layout import PageLayout
from rag_my_docs.structures.pdf_document import PdfDocument
from rag_my_docs.structures.pdf_page import PdfPage
from rag_my_docs.structures.raw_block import RawBlock
from rag_my_docs.structures.raw_table import RawTable
from rag_my_docs.structures.text_block import TextBlock


def helper_block(name: str, box: tuple[float, float, float, float]) -> RawBlock:
    return RawBlock(page=1, x0=box[0], y0=box[1], x1=box[2], y1=box[3], lines=(name,), font_size=10.0, bold_share=0.0)


def helper_table(box: tuple[float, float, float, float]) -> RawTable:
    return RawTable(
        page=1,
        x0=box[0],
        y0=box[1],
        x1=box[2],
        y1=box[3],
        header=("Region", "Sales"),
        rows=(("North", "120"), ("South", "95")),
    )


def helper_page(blocks: list[RawBlock], tables: list[RawTable]) -> PdfPage:
    return PdfPage(number=1, width=600.0, height=800.0, blocks=tuple(blocks), tables=tuple(tables), image_count=0)


def helper_names(items: list[RawBlock | RawTable]) -> list[str]:
    return [item.lines[0] if isinstance(item, RawBlock) else "TABLE" for item in items]


def test_blocks() -> None:
    first = PdfPage(
        number=1,
        width=600.0,
        height=800.0,
        blocks=(
            RawBlock(page=1, x0=50, y0=200, x1=550, y1=240, lines=("Second.",), font_size=10.0, bold_share=0.0),
            RawBlock(page=1, x0=50, y0=80, x1=550, y1=100, lines=("Title",), font_size=16.0, bold_share=1.0),
        ),
        tables=(),
        image_count=0,
    )
    second = PdfPage(
        number=2,
        width=600.0,
        height=800.0,
        blocks=(
            RawBlock(page=2, x0=50, y0=400, x1=550, y1=480, lines=("Next", "page."), font_size=10.0, bold_share=0.5),
        ),
        tables=(),
        image_count=0,
    )
    tested = PageLayout()
    result = tested.blocks(PdfDocument(pages=(first, second), outline=()))
    expected = [
        TextBlock(
            block_id=0,
            page=1,
            kind=BlockKind.PARAGRAPH,
            text="Title",
            font_size=16.0,
            bold_share=1.0,
            line_count=1,
            top=0.1,
            bottom=0.125,
        ),
        TextBlock(
            block_id=1,
            page=1,
            kind=BlockKind.PARAGRAPH,
            text="Second.",
            font_size=10.0,
            bold_share=0.0,
            line_count=1,
            top=0.25,
            bottom=0.3,
        ),
        TextBlock(
            block_id=2,
            page=2,
            kind=BlockKind.PARAGRAPH,
            text="Next page.",
            font_size=10.0,
            bold_share=0.5,
            line_count=2,
            top=0.5,
            bottom=0.6,
        ),
    ]
    assert result == expected


@pytest.mark.parametrize(
    "blocks, tables, expected",
    [
        pytest.param(
            [
                helper_block("Right 1", (320, 200, 550, 300)),
                helper_block("Left 2", (50, 310, 280, 400)),
                helper_block("Title", (50, 50, 550, 80)),
                helper_block("Left 1", (50, 200, 280, 300)),
                helper_block("Right 2", (320, 310, 550, 400)),
                helper_block("Full width", (50, 500, 550, 560)),
            ],
            [],
            ["Title", "Left 1", "Left 2", "Right 1", "Right 2", "Full width"],
            id="two-columns-between-spanning-blocks",
        ),
        pytest.param(
            [
                helper_block("Date", (450, 50, 550, 65)),
                helper_block("Para 1", (50, 100, 280, 200)),
                helper_block("Para 2", (50, 210, 280, 300)),
            ],
            [],
            ["Date", "Para 1", "Para 2"],
            id="right-aligned-line-is-not-a-column",
        ),
        pytest.param(
            [
                helper_block("Region Sales", (60, 300, 500, 315)),
                helper_block("After", (50, 400, 550, 420)),
                helper_block("Before", (50, 100, 550, 120)),
            ],
            [helper_table((50, 290, 550, 360))],
            ["Before", "TABLE", "After"],
            id="table-text-not-read-twice",
        ),
        pytest.param([], [], [], id="empty-page"),
    ],
)
def test_ordered(blocks: list[RawBlock], tables: list[RawTable], expected: list[str]) -> None:
    tested = PageLayout()
    result = helper_names(tested.ordered(helper_page(blocks, tables)))
    assert result == expected


@pytest.mark.parametrize(
    "band, expected",
    [
        pytest.param(
            [
                helper_block("L1", (50, 100, 280, 200)),
                helper_block("R1", (320, 100, 550, 200)),
                helper_block("L2", (50, 210, 280, 300)),
                helper_block("R2", (320, 210, 550, 300)),
            ],
            ["L1", "L2", "R1", "R2"],
            id="columns",
        ),
        pytest.param(
            [
                helper_block("L1", (50, 100, 280, 200)),
                helper_block("L2", (50, 210, 280, 300)),
                helper_block("R1", (320, 400, 550, 450)),
                helper_block("R2", (320, 460, 550, 500)),
            ],
            ["L1", "L2", "R1", "R2"],
            id="not-side-by-side-top-to-bottom",
        ),
        pytest.param(
            [helper_block("R1", (320, 100, 550, 200)), helper_block("L1", (50, 150, 280, 250))],
            ["R1", "L1"],
            id="one-per-side-top-to-bottom",
        ),
    ],
)
def test_band_order(band: list[RawBlock], expected: list[str]) -> None:
    tested = PageLayout
    result = helper_names(tested.band_order(list(band), 300.0, 30.0))
    assert result == expected


@pytest.mark.parametrize(
    "block, expected",
    [
        pytest.param(helper_block("inside", (60, 300, 200, 320)), True, id="inside"),
        pytest.param(helper_block("half", (0, 300, 100, 320)), True, id="exactly-half"),
        pytest.param(helper_block("mostly-out", (0, 300, 90, 320)), False, id="mostly-outside"),
        pytest.param(helper_block("away", (60, 600, 200, 620)), False, id="disjoint"),
        pytest.param(helper_block("flat", (60, 300, 200, 300)), False, id="zero-area"),
    ],
)
def test_is_in_table(block: RawBlock, expected: bool) -> None:
    tested = PageLayout
    result = tested.is_in_table(block, (helper_table((50, 290, 550, 360)),))
    assert result is expected


def test_text_block() -> None:
    page = helper_page([], [])
    tested = PageLayout()
    result = (
        tested.text_block(4, page, helper_table((50, 80, 550, 160))),
        tested.text_block(5, page, helper_block("Words", (50, 400, 550, 480))),
    )
    expected = (
        TextBlock(
            block_id=4,
            page=1,
            kind=BlockKind.TABLE,
            text="| Region | Sales |\n| North | 120 |\n| South | 95 |",
            font_size=0.0,
            bold_share=0.0,
            line_count=3,
            top=0.1,
            bottom=0.2,
        ),
        TextBlock(
            block_id=5,
            page=1,
            kind=BlockKind.PARAGRAPH,
            text="Words",
            font_size=10.0,
            bold_share=0.0,
            line_count=1,
            top=0.5,
            bottom=0.6,
        ),
    )
    assert result == expected


@pytest.mark.parametrize(
    "lines, expected",
    [
        pytest.param(("One wrapped", "line."), "One wrapped line.", id="wrapped"),
        pytest.param(
            ("Items:", "• first", "- second", "1. third", "(b) fourth", "2) fifth"),
            "Items:\n• first\n- second\n1. third\n(b) fourth\n2) fifth",
            id="list-items-keep-their-line",
        ),
        pytest.param(("Only",), "Only", id="single"),
        pytest.param(("Costs", "1.5 more"), "Costs 1.5 more", id="decimal-is-not-a-list-marker"),
    ],
)
def test_joined(lines: tuple[str, ...], expected: str) -> None:
    tested = PageLayout
    result = tested.joined(lines)
    assert result == expected


def test_table_text() -> None:
    tested = PageLayout
    result = tested.table_text(helper_table((0, 0, 1, 1)))
    expected = "| Region | Sales |\n| North | 120 |\n| South | 95 |"
    assert result == expected


@pytest.mark.parametrize(
    "cells, expected",
    [
        pytest.param(("a", "b"), "| a | b |", id="two"),
        pytest.param(("", "x"), "|  | x |", id="empty-cell"),
        pytest.param(("solo",), "| solo |", id="one"),
    ],
)
def test_row_text(cells: tuple[str, ...], expected: str) -> None:
    tested = PageLayout
    result = tested.row_text(cells)
    assert result == expected
