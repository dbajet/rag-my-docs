from __future__ import annotations

from rag_my_docs.constants.link_purpose import LinkPurpose


def test_enum() -> None:
    tested = LinkPurpose
    expected = {"SIGN_IN": "sign_in"}
    assert {member.name: member.value for member in tested} == expected
