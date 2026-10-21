from __future__ import annotations

from typing import Any

from rag_my_docs.structures.search_hit import SearchHit
from tests.conftest import is_namedtuple


def helper_instance() -> SearchHit:
    return SearchHit(
        rank=1,
        chunk_id=12,
        chunk_position=2,
        document_id=3,
        document_name="report.pdf",
        section_id=7,
        section_path="2. Results › 2.1 Details",
        start_page=2,
        end_page=3,
        text="Sales grew in both regions.",
        prefix_chars=0,
        similarity=0.8125,
    )


def helper_dict() -> dict[str, Any]:
    return {
        "rank": 1,
        "chunk_id": 12,
        "chunk_position": 2,
        "document_id": 3,
        "document_name": "report.pdf",
        "section_id": 7,
        "section_path": "2. Results › 2.1 Details",
        "start_page": 2,
        "end_page": 3,
        "text": "Sales grew in both regions.",
        "prefix_chars": 0,
        "similarity": 0.8125,
    }


def test_class() -> None:
    tested = SearchHit
    fields = {
        "rank": "int",
        "chunk_id": "int",
        "chunk_position": "int",
        "document_id": "int",
        "document_name": "str",
        "section_id": "int",
        "section_path": "str",
        "start_page": "int",
        "end_page": "int",
        "text": "str",
        "prefix_chars": "int",
        "similarity": "float",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = helper_instance()
    result = tested.to_dict()
    expected = helper_dict()
    assert result == expected


def test_from_dict() -> None:
    tested = SearchHit
    result = tested.from_dict(helper_dict() | {"rank": "1", "similarity": "0.8125"})
    expected = helper_instance()
    assert result == expected
