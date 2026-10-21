from __future__ import annotations

import pytest

from rag_my_docs.structures.pdf_page import PdfPage
from rag_my_docs.structures.raw_block import RawBlock
from rag_my_docs.structures.raw_table import RawTable
from tests.conftest import is_namedtuple


def helper_block(lines: tuple[str, ...]) -> RawBlock:
    return RawBlock(page=1, x0=72.0, y0=100.0, x1=500.0, y1=120.0, lines=lines, font_size=10.0, bold_share=0.0)


def helper_table() -> RawTable:
    return RawTable(page=1, x0=72.0, y0=300.0, x1=522.0, y1=340.0, header=("A",), rows=(("1",),))


def helper_instance(blocks: tuple[RawBlock, ...]) -> PdfPage:
    return PdfPage(number=1, width=595.0, height=842.0, blocks=blocks, tables=(helper_table(),), image_count=2)


def test_class() -> None:
    tested = PdfPage
    fields = {
        "number": "int",
        "width": "float",
        "height": "float",
        "blocks": "tuple[RawBlock, ...]",
        "tables": "tuple[RawTable, ...]",
        "image_count": "int",
    }
    assert is_namedtuple(tested, fields)


@pytest.mark.parametrize(
    "blocks, expected",
    [
        pytest.param((), 0, id="no-block"),
        pytest.param((helper_block(("Hello there",)),), 11, id="one-line"),
        pytest.param((helper_block(("  ab ", "cd")), helper_block(("e",))), 5, id="trimmed-lines-of-two-blocks"),
    ],
)
def test_char_count(blocks: tuple[RawBlock, ...], expected: int) -> None:
    tested = helper_instance(blocks)
    result = tested.char_count()
    assert result == expected


def test_to_dict() -> None:
    tested = helper_instance((helper_block(("Hi",)),))
    result = tested.to_dict()
    expected = {
        "number": 1,
        "width": 595.0,
        "height": 842.0,
        "blocks": [
            {
                "page": 1,
                "x0": 72.0,
                "y0": 100.0,
                "x1": 500.0,
                "y1": 120.0,
                "lines": ["Hi"],
                "font_size": 10.0,
                "bold_share": 0.0,
            }
        ],
        "tables": [{"page": 1, "x0": 72.0, "y0": 300.0, "x1": 522.0, "y1": 340.0, "header": ["A"], "rows": [["1"]]}],
        "image_count": 2,
    }
    assert result == expected


def test_from_dict() -> None:
    tested = PdfPage
    result = tested.from_dict(
        {
            "number": "1",
            "width": 595,
            "height": "842",
            "blocks": [
                {
                    "page": 1,
                    "x0": 72,
                    "y0": 100,
                    "x1": 500,
                    "y1": 120,
                    "lines": ["Hi"],
                    "font_size": 10,
                    "bold_share": 0,
                }
            ],
            "tables": [{"page": 1, "x0": 72, "y0": 300, "x1": 522, "y1": 340, "header": ["A"], "rows": [["1"]]}],
            "image_count": "2",
        }
    )
    expected = helper_instance((helper_block(("Hi",)),))
    assert result == expected
