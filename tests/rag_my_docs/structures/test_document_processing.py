from __future__ import annotations

from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.structures.document_processing import DocumentProcessing
from tests.conftest import is_namedtuple


def helper_instance(page_count: int | None) -> DocumentProcessing:
    return DocumentProcessing(
        document_id=3,
        name="report.pdf",
        media_type=MediaType.PDF,
        status=ProcessingStatus.COMPLETED,
        error="",
        page_count=page_count,
        chunking_version="sentences-v1 target=200 max=350 min=75 overlap=1",
        processed_at="2026-10-19T12:00:00Z",
        report={"tables": 1},
        index_status=IndexStatus.INDEXED,
        index_model="BAAI/bge-small-en-v1.5",
        index_error="",
        indexed_at="2026-10-19T12:01:00Z",
        index_report={"chunks": 5, "embedded": 5},
        chunk_count=5,
        indexed_count=5,
    )


def test_class() -> None:
    tested = DocumentProcessing
    fields = {
        "document_id": "int",
        "name": "str",
        "media_type": "MediaType",
        "status": "ProcessingStatus",
        "error": "str",
        "page_count": "int | None",
        "chunking_version": "str",
        "processed_at": "str",
        "report": "dict[str, Any]",
        "index_status": "IndexStatus",
        "index_model": "str",
        "index_error": "str",
        "indexed_at": "str",
        "index_report": "dict[str, Any]",
        "chunk_count": "int",
        "indexed_count": "int",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = helper_instance(4)
    result = tested.to_dict()
    expected = {
        "document_id": 3,
        "name": "report.pdf",
        "media_type": "application/pdf",
        "status": "completed",
        "error": "",
        "page_count": 4,
        "chunking_version": "sentences-v1 target=200 max=350 min=75 overlap=1",
        "processed_at": "2026-10-19T12:00:00Z",
        "report": {"tables": 1},
        "index_status": "indexed",
        "index_model": "BAAI/bge-small-en-v1.5",
        "index_error": "",
        "indexed_at": "2026-10-19T12:01:00Z",
        "index_report": {"chunks": 5, "embedded": 5},
        "chunk_count": 5,
        "indexed_count": 5,
    }
    assert result == expected


def test_from_dict() -> None:
    tested = DocumentProcessing
    data = {
        "document_id": "3",
        "name": "report.pdf",
        "media_type": "application/pdf",
        "status": "completed",
        "error": "",
        "page_count": "4",
        "chunking_version": "sentences-v1 target=200 max=350 min=75 overlap=1",
        "processed_at": "2026-10-19T12:00:00Z",
        "report": {"tables": 1},
        "index_status": "indexed",
        "index_model": "BAAI/bge-small-en-v1.5",
        "index_error": "",
        "indexed_at": "2026-10-19T12:01:00Z",
        "index_report": {"chunks": 5, "embedded": 5},
        "chunk_count": "5",
        "indexed_count": "5",
    }
    result = (tested.from_dict(data), tested.from_dict(data | {"page_count": None}))
    expected = (helper_instance(4), helper_instance(None))
    assert result == expected
