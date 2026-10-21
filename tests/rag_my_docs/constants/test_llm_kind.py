from __future__ import annotations

from rag_my_docs.constants.llm_kind import LlmKind


def test_enum() -> None:
    tested = LlmKind
    expected = {"CLAUDE": "claude", "OPENAI": "openai"}
    assert {member.name: member.value for member in tested} == expected
