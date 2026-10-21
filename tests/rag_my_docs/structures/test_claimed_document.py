from __future__ import annotations

from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.structures.claimed_document import ClaimedDocument
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = ClaimedDocument
    fields = {"document_id": "int", "run": "int", "media_type": "MediaType", "content": "bytes"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = ClaimedDocument(document_id=3, run=2, media_type=MediaType.PDF, content=b"%PDF-1.7")
    result = tested.to_dict()
    expected = {"document_id": 3, "run": 2, "media_type": "application/pdf", "content": b"%PDF-1.7"}
    assert result == expected


def test_from_dict() -> None:
    tested = ClaimedDocument
    result = tested.from_dict(
        {"document_id": "3", "run": "2", "media_type": "image/png", "content": memoryview(b"\x89PNG")}
    )
    expected = ClaimedDocument(document_id=3, run=2, media_type=MediaType.PNG, content=b"\x89PNG")
    assert result == expected
