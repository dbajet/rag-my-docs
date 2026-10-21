from __future__ import annotations

import pytest

from rag_my_docs.structures.paragraph import Paragraph
from rag_my_docs.structures.paragraph_segment import ParagraphSegment
from rag_my_docs.structures.source_span import SourceSpan
from tests.conftest import is_namedtuple


def helper_across_pages() -> Paragraph:
    """ "Alpha beta" ends page 1 (block 4), "gamma delta." starts page 2 (block 5),
    joined by one space: paragraph characters 11 on are block 5's from 0 on."""
    return Paragraph(
        text="Alpha beta gamma delta.",
        segments=(
            ParagraphSegment(block_id=4, page=1, offset=0, length=10),
            ParagraphSegment(block_id=5, page=2, offset=11, length=12),
        ),
    )


def test_class() -> None:
    tested = Paragraph
    fields = {"text": "str", "segments": "tuple[ParagraphSegment, ...]"}
    assert is_namedtuple(tested, fields)


@pytest.mark.parametrize(
    "start, end, expected",
    [
        pytest.param(0, 10, (SourceSpan(block_id=4, start=0, end=10),), id="first-block-whole"),
        pytest.param(
            6,
            16,
            (SourceSpan(block_id=4, start=6, end=10), SourceSpan(block_id=5, start=0, end=5)),
            id="crossing-the-page-break",
        ),
        pytest.param(10, 11, (), id="only-the-joining-space"),
        pytest.param(11, 23, (SourceSpan(block_id=5, start=0, end=12),), id="second-block-whole"),
    ],
)
def test_spans(start: int, end: int, expected: tuple[SourceSpan, ...]) -> None:
    tested = helper_across_pages()
    result = tested.spans(start, end)
    assert result == expected


@pytest.mark.parametrize(
    "start, end, expected",
    [
        pytest.param(0, 5, (1, 1), id="page-one"),
        pytest.param(6, 16, (1, 2), id="crossing"),
        pytest.param(12, 23, (2, 2), id="page-two"),
    ],
)
def test_pages(start: int, end: int, expected: tuple[int, int]) -> None:
    tested = helper_across_pages()
    result = tested.pages(start, end)
    assert result == expected


def test_to_dict() -> None:
    tested = helper_across_pages()
    result = tested.to_dict()
    expected = {
        "text": "Alpha beta gamma delta.",
        "segments": [
            {"block_id": 4, "page": 1, "offset": 0, "length": 10},
            {"block_id": 5, "page": 2, "offset": 11, "length": 12},
        ],
    }
    assert result == expected


def test_from_dict() -> None:
    tested = Paragraph
    result = tested.from_dict(
        {
            "text": "Alpha beta gamma delta.",
            "segments": [
                {"block_id": "4", "page": "1", "offset": "0", "length": "10"},
                {"block_id": 5, "page": 2, "offset": 11, "length": 12},
            ],
        }
    )
    expected = helper_across_pages()
    assert result == expected
