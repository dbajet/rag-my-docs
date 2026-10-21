from __future__ import annotations

from rag_my_docs.structures.source_span import SourceSpan
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = SourceSpan
    fields = {"block_id": "int", "start": "int", "end": "int"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = SourceSpan(block_id=4, start=0, end=12)
    result = tested.to_dict()
    expected = {"block_id": 4, "start": 0, "end": 12}
    assert result == expected


def test_from_dict() -> None:
    tested = SourceSpan
    result = tested.from_dict({"block_id": "4", "start": "0", "end": 12})
    expected = SourceSpan(block_id=4, start=0, end=12)
    assert result == expected
