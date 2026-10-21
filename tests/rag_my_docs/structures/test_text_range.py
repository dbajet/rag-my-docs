from __future__ import annotations

from rag_my_docs.structures.text_range import TextRange
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = TextRange
    fields = {"start": "int", "end": "int"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = TextRange(start=3, end=9)
    result = tested.to_dict()
    expected = {"start": 3, "end": 9}
    assert result == expected


def test_from_dict() -> None:
    tested = TextRange
    result = tested.from_dict({"start": "3", "end": 9})
    expected = TextRange(start=3, end=9)
    assert result == expected
