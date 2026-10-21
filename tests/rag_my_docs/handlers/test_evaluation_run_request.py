from __future__ import annotations

from pydantic_core import PydanticUndefined

from rag_my_docs.handlers.evaluation_run_request import EvaluationRunRequest


def test_class() -> None:
    tested = EvaluationRunRequest
    result = {name: (field.annotation, field.default) for name, field in tested.model_fields.items()}
    expected = {"index": (int, PydanticUndefined), "top_k": (int, 5)}
    assert result == expected
