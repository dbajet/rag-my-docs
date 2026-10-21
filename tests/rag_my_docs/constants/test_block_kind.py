from __future__ import annotations

from rag_my_docs.constants.block_kind import BlockKind


def test_enum() -> None:
    tested = BlockKind
    expected = {
        "PARAGRAPH": "paragraph",
        "HEADING": "heading",
        "TABLE": "table",
        "HEADER": "header",
        "FOOTER": "footer",
    }
    assert {member.name: member.value for member in tested} == expected
