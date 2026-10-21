from __future__ import annotations

from rag_my_docs.structures.claimed_indexing import ClaimedIndexing
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = ClaimedIndexing
    fields = {"document_id": "int", "run": "int", "model": "str"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = ClaimedIndexing(document_id=3, run=2, model="BAAI/bge-small-en-v1.5")
    result = tested.to_dict()
    expected = {"document_id": 3, "run": 2, "model": "BAAI/bge-small-en-v1.5"}
    assert result == expected


def test_from_dict() -> None:
    tested = ClaimedIndexing
    result = tested.from_dict({"document_id": "3", "run": "2", "model": "BAAI/bge-small-en-v1.5"})
    expected = ClaimedIndexing(document_id=3, run=2, model="BAAI/bge-small-en-v1.5")
    assert result == expected
