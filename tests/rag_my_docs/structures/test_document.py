from __future__ import annotations

from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.structures.document import Document
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = Document
    fields = {
        "document_id": "int",
        "user_id": "int",
        "name": "str",
        "media_type": "MediaType",
        "size_bytes": "int",
        "created_at": "str",
        "processing_status": "ProcessingStatus",
        "index_status": "IndexStatus",
        "chunk_count": "int",
        "indexed_count": "int",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = Document(
        document_id=5,
        user_id=7,
        name="report.pdf",
        media_type=MediaType.PDF,
        size_bytes=1234,
        created_at="2026-10-19T12:00:00Z",
        processing_status=ProcessingStatus.COMPLETED,
        index_status=IndexStatus.INDEXED,
        chunk_count=12,
        indexed_count=12,
    )
    result = tested.to_dict()
    expected = {
        "document_id": 5,
        "user_id": 7,
        "name": "report.pdf",
        "media_type": "application/pdf",
        "size_bytes": 1234,
        "created_at": "2026-10-19T12:00:00Z",
        "processing_status": "completed",
        "index_status": "indexed",
        "chunk_count": 12,
        "indexed_count": 12,
    }
    assert result == expected


def test_from_dict() -> None:
    tested = Document
    result = tested.from_dict(
        {
            "document_id": "5",
            "user_id": "7",
            "name": "notes.md",
            "media_type": "text/markdown",
            "size_bytes": "12",
            "created_at": "2026-10-19T12:00:00Z",
            "processing_status": "unsupported",
            "index_status": "not_indexed",
            "chunk_count": "0",
            "indexed_count": "0",
        }
    )
    expected = Document(
        document_id=5,
        user_id=7,
        name="notes.md",
        media_type=MediaType.MARKDOWN,
        size_bytes=12,
        created_at="2026-10-19T12:00:00Z",
        processing_status=ProcessingStatus.UNSUPPORTED,
        index_status=IndexStatus.NOT_INDEXED,
        chunk_count=0,
        indexed_count=0,
    )
    assert result == expected
