from __future__ import annotations

from rag_my_docs.constants.media_type import MediaType


def test_enum() -> None:
    tested = MediaType
    expected = {
        "PDF": "application/pdf",
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "TEXT": "text/plain",
        "MARKDOWN": "text/markdown",
        "CSV": "text/csv",
    }
    assert {member.name: member.value for member in tested} == expected
