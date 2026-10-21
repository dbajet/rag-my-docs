from __future__ import annotations

from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.structures.chunk import Chunk
from rag_my_docs.structures.source_span import SourceSpan
from tests.conftest import is_namedtuple


def helper_instance() -> Chunk:
    return Chunk(
        position=3,
        section_position=1,
        kind=ChunkKind.PROSE,
        text="Before. After here.",
        prefix_chars=8,
        token_count=5,
        start_page=1,
        end_page=2,
        spans=(SourceSpan(block_id=4, start=210, end=221), SourceSpan(block_id=7, start=0, end=5)),
    )


def test_class() -> None:
    tested = Chunk
    fields = {
        "position": "int",
        "section_position": "int",
        "kind": "ChunkKind",
        "text": "str",
        "prefix_chars": "int",
        "token_count": "int",
        "start_page": "int",
        "end_page": "int",
        "spans": "tuple[SourceSpan, ...]",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = helper_instance()
    result = tested.to_dict()
    expected = {
        "position": 3,
        "section_position": 1,
        "kind": "prose",
        "text": "Before. After here.",
        "prefix_chars": 8,
        "token_count": 5,
        "start_page": 1,
        "end_page": 2,
        "spans": [{"block_id": 4, "start": 210, "end": 221}, {"block_id": 7, "start": 0, "end": 5}],
    }
    assert result == expected


def test_from_dict() -> None:
    tested = Chunk
    result = tested.from_dict(
        {
            "position": "3",
            "section_position": "1",
            "kind": "prose",
            "text": "Before. After here.",
            "prefix_chars": "8",
            "token_count": "5",
            "start_page": "1",
            "end_page": "2",
            "spans": [{"block_id": 4, "start": 210, "end": 221}, {"block_id": 7, "start": 0, "end": 5}],
        }
    )
    expected = helper_instance()
    assert result == expected
