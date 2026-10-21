from __future__ import annotations

from rag_my_docs.constants.chunk_kind import ChunkKind


def test_enum() -> None:
    tested = ChunkKind
    expected = {
        "PROSE": "prose",
        "TABLE": "table",
    }
    assert {member.name: member.value for member in tested} == expected
