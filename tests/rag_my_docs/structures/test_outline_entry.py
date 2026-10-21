from __future__ import annotations

from rag_my_docs.structures.outline_entry import OutlineEntry
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = OutlineEntry
    fields = {"level": "int", "title": "str", "page": "int"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = OutlineEntry(level=2, title="1.1 Background", page=3)
    result = tested.to_dict()
    expected = {"level": 2, "title": "1.1 Background", "page": 3}
    assert result == expected


def test_from_dict() -> None:
    tested = OutlineEntry
    result = tested.from_dict({"level": "1", "title": "Introduction", "page": "1"})
    expected = OutlineEntry(level=1, title="Introduction", page=1)
    assert result == expected
