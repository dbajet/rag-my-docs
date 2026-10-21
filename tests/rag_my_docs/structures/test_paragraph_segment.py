from __future__ import annotations

from rag_my_docs.structures.paragraph_segment import ParagraphSegment
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = ParagraphSegment
    fields = {"block_id": "int", "page": "int", "offset": "int", "length": "int"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = ParagraphSegment(block_id=5, page=2, offset=12, length=30)
    result = tested.to_dict()
    expected = {"block_id": 5, "page": 2, "offset": 12, "length": 30}
    assert result == expected


def test_from_dict() -> None:
    tested = ParagraphSegment
    result = tested.from_dict({"block_id": "5", "page": "2", "offset": "12", "length": 30})
    expected = ParagraphSegment(block_id=5, page=2, offset=12, length=30)
    assert result == expected
