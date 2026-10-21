from __future__ import annotations

from rag_my_docs.constants.index_status import IndexStatus


def test_enum() -> None:
    tested = IndexStatus
    expected = {
        "NOT_INDEXED": "not_indexed",
        "PENDING": "pending",
        "INDEXING": "indexing",
        "INDEXED": "indexed",
        "FAILED": "failed",
    }
    assert {member.name: member.value for member in tested} == expected
