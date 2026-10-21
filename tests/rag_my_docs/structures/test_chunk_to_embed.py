from __future__ import annotations

from rag_my_docs.structures.chunk_to_embed import ChunkToEmbed
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = ChunkToEmbed
    fields = {"chunk_id": "int", "text": "str"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = ChunkToEmbed(chunk_id=4, text="Sales grew.")
    result = tested.to_dict()
    expected = {"chunk_id": 4, "text": "Sales grew."}
    assert result == expected


def test_from_dict() -> None:
    tested = ChunkToEmbed
    result = tested.from_dict({"chunk_id": "4", "text": "Sales grew."})
    expected = ChunkToEmbed(chunk_id=4, text="Sales grew.")
    assert result == expected
