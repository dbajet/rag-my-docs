from __future__ import annotations

from pydantic_core import PydanticUndefined

from rag_my_docs.constants.llm_kind import LlmKind
from rag_my_docs.handlers.llm_models_request import LlmModelsRequest


def test_class() -> None:
    tested = LlmModelsRequest
    result = {name: (field.annotation, field.default) for name, field in tested.model_fields.items()}
    expected = {
        "connection_id": (int, 0),
        "kind": (LlmKind, PydanticUndefined),
        "base_url": (str, ""),
        "api_key": (str, ""),
    }
    assert result == expected
