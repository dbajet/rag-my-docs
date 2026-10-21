from __future__ import annotations

from rag_my_docs.structures.raw_block import RawBlock
from tests.conftest import is_namedtuple


def helper_instance() -> RawBlock:
    return RawBlock(
        page=1,
        x0=72.0,
        y0=95.5,
        x1=169.0,
        y1=114.0,
        lines=("1. Introduction",),
        font_size=14.0,
        bold_share=1.0,
    )


def test_class() -> None:
    tested = RawBlock
    fields = {
        "page": "int",
        "x0": "float",
        "y0": "float",
        "x1": "float",
        "y1": "float",
        "lines": "tuple[str, ...]",
        "font_size": "float",
        "bold_share": "float",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = helper_instance()
    result = tested.to_dict()
    expected = {
        "page": 1,
        "x0": 72.0,
        "y0": 95.5,
        "x1": 169.0,
        "y1": 114.0,
        "lines": ["1. Introduction"],
        "font_size": 14.0,
        "bold_share": 1.0,
    }
    assert result == expected


def test_from_dict() -> None:
    tested = RawBlock
    result = tested.from_dict(
        {
            "page": "1",
            "x0": 72,
            "y0": "95.5",
            "x1": 169,
            "y1": 114,
            "lines": ["1. Introduction"],
            "font_size": "14",
            "bold_share": 1,
        }
    )
    expected = helper_instance()
    assert result == expected
