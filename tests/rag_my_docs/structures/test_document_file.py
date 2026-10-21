from __future__ import annotations

from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.structures.document_file import DocumentFile
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = DocumentFile
    fields = {"name": "str", "media_type": "MediaType", "content": "bytes"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = DocumentFile(name="photo.png", media_type=MediaType.PNG, content=b"\x89PNG")
    result = tested.to_dict()
    expected = {"name": "photo.png", "media_type": "image/png", "content": b"\x89PNG"}
    assert result == expected


def test_from_dict() -> None:
    tested = DocumentFile
    result = tested.from_dict({"name": "a.csv", "media_type": "text/csv", "content": memoryview(b"a,b")})
    expected = DocumentFile(name="a.csv", media_type=MediaType.CSV, content=b"a,b")
    assert result == expected
