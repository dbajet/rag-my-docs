from __future__ import annotations

from pydantic_core import PydanticUndefined

from rag_my_docs.handlers.answer_request import AnswerRequest


def test_class() -> None:
    tested = AnswerRequest
    result = {name: (field.annotation, field.default) for name, field in tested.model_fields.items()}
    expected = {"question": (str, PydanticUndefined), "top_k": (int, 5)}
    assert result == expected
