from __future__ import annotations

from rag_my_docs.libraries.email_composer import EmailComposer


def test_sign_in_subject() -> None:
    tested = EmailComposer()
    result = tested.sign_in_subject()
    expected = "Your RAG My Docs sign-in link"
    assert result == expected


def test_sign_in_lines() -> None:
    tested = EmailComposer()
    result = tested.sign_in_lines("Ada", "https://docs.example.com/auth/link?token=tok123")
    expected = [
        "Hello Ada,",
        "",
        "To sign in to RAG My Docs, open this link:",
        "https://docs.example.com/auth/link?token=tok123",
        "",
        "It works once, for 15 minutes.",
        "If you did not ask for it, ignore this message.",
    ]
    assert result == expected
    exp_link = "https://docs.example.com/auth/link?token=tok123"
    assert exp_link in result
