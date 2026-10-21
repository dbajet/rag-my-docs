from __future__ import annotations

from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.structures.source_span import SourceSpan
from rag_my_docs.structures.stored_chunk import StoredChunk
from tests.conftest import is_namedtuple


def helper_instance() -> StoredChunk:
    return StoredChunk(
        chunk_id=21,
        section_id=11,
        position=0,
        kind=ChunkKind.TABLE,
        text="| A | B |\n| 1 | 2 |",
        prefix_chars=0,
        token_count=12,
        start_page=2,
        end_page=2,
        spans=(SourceSpan(block_id=11, start=0, end=19),),
        strategy="sentences-v1 target=200 max=350 min=75 overlap=1",
        embedding_model="BAAI/bge-small-en-v1.5",
    )


def test_class() -> None:
    tested = StoredChunk
    fields = {
        "chunk_id": "int",
        "section_id": "int",
        "position": "int",
        "kind": "ChunkKind",
        "text": "str",
        "prefix_chars": "int",
        "token_count": "int",
        "start_page": "int",
        "end_page": "int",
        "spans": "tuple[SourceSpan, ...]",
        "strategy": "str",
        "embedding_model": "str",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = helper_instance()
    result = tested.to_dict()
    expected = {
        "chunk_id": 21,
        "section_id": 11,
        "position": 0,
        "kind": "table",
        "text": "| A | B |\n| 1 | 2 |",
        "prefix_chars": 0,
        "token_count": 12,
        "start_page": 2,
        "end_page": 2,
        "spans": [{"block_id": 11, "start": 0, "end": 19}],
        "strategy": "sentences-v1 target=200 max=350 min=75 overlap=1",
        "embedding_model": "BAAI/bge-small-en-v1.5",
    }
    assert result == expected


def test_from_dict() -> None:
    tested = StoredChunk
    result = tested.from_dict(
        {
            "chunk_id": "21",
            "section_id": "11",
            "position": "0",
            "kind": "table",
            "text": "| A | B |\n| 1 | 2 |",
            "prefix_chars": "0",
            "token_count": "12",
            "start_page": "2",
            "end_page": "2",
            "spans": [{"block_id": "11", "start": "0", "end": "19"}],
            "strategy": "sentences-v1 target=200 max=350 min=75 overlap=1",
            "embedding_model": "BAAI/bge-small-en-v1.5",
        }
    )
    expected = helper_instance()
    assert result == expected
