from __future__ import annotations

from rag_my_docs.structures.embedding_carry import EmbeddingCarry
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = EmbeddingCarry
    fields = {"vector": "str", "model": "str"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = EmbeddingCarry(vector="[0.6,0.8]", model="BAAI/bge-small-en-v1.5")
    result = tested.to_dict()
    expected = {"vector": "[0.6,0.8]", "model": "BAAI/bge-small-en-v1.5"}
    assert result == expected


def test_from_dict() -> None:
    tested = EmbeddingCarry
    result = tested.from_dict({"vector": "[0.6,0.8]", "model": "BAAI/bge-small-en-v1.5"})
    expected = EmbeddingCarry(vector="[0.6,0.8]", model="BAAI/bge-small-en-v1.5")
    assert result == expected
