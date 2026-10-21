from __future__ import annotations

from rag_my_docs.handlers.llm_routing_request import LlmRoutingRequest


def test_class() -> None:
    tested = LlmRoutingRequest
    result = {name: (field.annotation, field.default) for name, field in tested.model_fields.items()}
    expected = {"answers": (int | None, None)}
    assert result == expected
