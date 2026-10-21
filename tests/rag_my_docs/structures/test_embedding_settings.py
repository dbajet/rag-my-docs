from __future__ import annotations

from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = EmbeddingSettings
    fields = {"model": "str", "query_prefix": "str", "batch_size": "int"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = EmbeddingSettings(model="BAAI/bge-small-en-v1.5", query_prefix="Query: ", batch_size=16)
    result = tested.to_dict()
    expected = {"model": "BAAI/bge-small-en-v1.5", "query_prefix": "Query: ", "batch_size": 16}
    assert result == expected


def test_from_dict() -> None:
    tested = EmbeddingSettings
    result = tested.from_dict({"model": "BAAI/bge-small-en-v1.5", "query_prefix": "", "batch_size": "16"})
    expected = EmbeddingSettings(model="BAAI/bge-small-en-v1.5", query_prefix="", batch_size=16)
    assert result == expected
