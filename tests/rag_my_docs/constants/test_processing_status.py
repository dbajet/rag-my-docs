from __future__ import annotations

from rag_my_docs.constants.processing_status import ProcessingStatus


def test_enum() -> None:
    tested = ProcessingStatus
    expected = {
        "PENDING": "pending",
        "PROCESSING": "processing",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "NO_TEXT": "no_text",
        "UNSUPPORTED": "unsupported",
    }
    assert {member.name: member.value for member in tested} == expected
