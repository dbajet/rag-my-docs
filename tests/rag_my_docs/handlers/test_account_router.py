from __future__ import annotations

from unittest.mock import call, patch

from starlette.requests import Request

from rag_my_docs.commands.session_command import SessionCommand
from rag_my_docs.handlers.account_router import AccountRouter
from rag_my_docs.handlers.session_guard import SessionGuard
from rag_my_docs.libraries.blind_index import BlindIndex
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.session_store import SessionStore
from rag_my_docs.libraries.user_store import UserStore
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.settings import Settings
from rag_my_docs.structures.user_account import UserAccount

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="


def helper_guard() -> SessionGuard:
    settings = Settings(
        environment="dev",
        database_url="postgresql://u:p@h/d",
        app_version="v",
        build_time="b",
        app_commit="c",
        app_base_url="https://docs.example.com",
        data_key=KEY,
        smtp_host="",
        smtp_port=587,
        smtp_user="",
        smtp_password="",
        email_from="",
        bootstrap_admin_email="",
        chunking=ChunkingSettings(target_tokens=200, max_tokens=350, min_tokens=75, overlap_sentences=1),
        embedding=EmbeddingSettings(model="BAAI/bge-small-en-v1.5", query_prefix="Q: ", batch_size=32),
    )
    database = Database(settings.database_url)
    users = UserStore(database, SecretSealer(KEY), BlindIndex(KEY))
    return SessionGuard(settings, SessionCommand(users, SessionStore(database)))


def test___init__() -> None:
    guard = helper_guard()
    tested = AccountRouter(guard)
    result = tested._guard
    expected = guard
    assert result == expected


def test_router() -> None:
    tested = AccountRouter(helper_guard())
    result = [(route.path, tuple(sorted(route.methods)), route.name) for route in tested.router().routes]  # type: ignore[attr-defined]
    expected = [("/api/account", ("GET",), "account")]
    assert result == expected


def test_account() -> None:
    tested = AccountRouter(helper_guard())
    request = Request({"type": "http", "method": "GET", "path": "/api/account", "headers": []})
    user = UserAccount(user_id=7, email="owner@example.com", display_name="owner", is_disabled=False)
    with patch.object(SessionGuard, "user", side_effect=[user]) as mock_user:
        result = tested.account(request)
    expected = {"user_id": 7, "email": "owner@example.com", "display_name": "owner", "is_disabled": False}
    assert result == expected
    exp_calls = [call(request)]
    assert mock_user.mock_calls == exp_calls
