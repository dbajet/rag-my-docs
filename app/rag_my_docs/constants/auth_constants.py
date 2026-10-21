from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _AuthConstants:
    cookie_name: str = "rag_my_docs_session"
    session_days: int = 30
    # Long enough to find the email, short enough that a forwarded one goes stale.
    magic_link_minutes: int = 15
    token_bytes: int = 32
    # The HMAC label deriving the blind-index key from the data key, so the two are never the same bytes.
    blind_index_label: str = "rag_my_docs email blind index v1"
    link_path: str = "/auth/link"
    # One answer for every sign-in request, whether or not the address has an account.
    link_requested: str = "If this address has access, a sign-in link is on its way."
    not_signed_in: str = "Please sign in."
    link_refused: str = "This sign-in link has expired or was already used. Ask for a new one."
    origin_refused: str = "This request did not come from RAG My Docs."
    unsafe_methods: tuple[str, ...] = ("POST", "PUT", "PATCH", "DELETE")


AuthConstants = _AuthConstants()
