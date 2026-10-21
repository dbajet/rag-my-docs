from __future__ import annotations

from rag_my_docs.constants.section_source import SectionSource


def test_enum() -> None:
    tested = SectionSource
    expected = {
        "BOOKMARK": "bookmark",
        "NUMBERED": "numbered",
        "LAYOUT": "layout",
        "FALLBACK": "fallback",
    }
    assert {member.name: member.value for member in tested} == expected
