from __future__ import annotations

from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.structures.text_block import TextBlock
from tests.conftest import is_namedtuple


def helper_instance() -> TextBlock:
    return TextBlock(
        block_id=4,
        page=2,
        kind=BlockKind.HEADING,
        text="2. Results",
        font_size=13.0,
        bold_share=1.0,
        line_count=1,
        top=0.15,
        bottom=0.17,
    )


def test_class() -> None:
    tested = TextBlock
    fields = {
        "block_id": "int",
        "page": "int",
        "kind": "BlockKind",
        "text": "str",
        "font_size": "float",
        "bold_share": "float",
        "line_count": "int",
        "top": "float",
        "bottom": "float",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = helper_instance()
    result = tested.to_dict()
    expected = {
        "block_id": 4,
        "page": 2,
        "kind": "heading",
        "text": "2. Results",
        "font_size": 13.0,
        "bold_share": 1.0,
        "line_count": 1,
        "top": 0.15,
        "bottom": 0.17,
    }
    assert result == expected


def test_from_dict() -> None:
    tested = TextBlock
    result = tested.from_dict(
        {
            "block_id": "4",
            "page": "2",
            "kind": "heading",
            "text": "2. Results",
            "font_size": 13,
            "bold_share": "1",
            "line_count": "1",
            "top": "0.15",
            "bottom": 0.17,
        }
    )
    expected = helper_instance()
    assert result == expected
