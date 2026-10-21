from __future__ import annotations

from rag_my_docs.handlers.index_request import IndexRequest


def test_class() -> None:
    tested = IndexRequest
    result = {name: (field.annotation, field.default) for name, field in tested.model_fields.items()}
    expected = {"rebuild": (bool, False)}
    assert result == expected
