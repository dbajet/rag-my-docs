from __future__ import annotations

from unittest.mock import call, patch

from fastapi.responses import JSONResponse
from starlette.requests import Request

from rag_my_docs.commands.session_command import SessionCommand
from rag_my_docs.commands.sign_in_command import SignInCommand
from rag_my_docs.handlers.auth_router import AuthRouter
from rag_my_docs.handlers.session_guard import SessionGuard
from rag_my_docs.handlers.sign_in_request import SignInRequest
from rag_my_docs.handlers.token_request import TokenRequest
from rag_my_docs.libraries.alerter import Alerter
from rag_my_docs.libraries.blind_index import BlindIndex
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.email_box import EmailBox
from rag_my_docs.libraries.email_composer import EmailComposer
from rag_my_docs.libraries.job_queue import JobQueue
from rag_my_docs.libraries.magic_link_store import MagicLinkStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.session_store import SessionStore
from rag_my_docs.libraries.user_store import UserStore
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.settings import Settings


def helper_settings() -> Settings:
    return Settings(
        environment="dev",
        database_url="postgresql://u:p@h/d",
        app_version="v",
        build_time="b",
        app_commit="c",
        app_base_url="https://docs.example.com",
        data_key="Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI=",
        smtp_host="",
        smtp_port=587,
        smtp_user="",
        smtp_password="",
        email_from="",
        bootstrap_admin_email="",
        chunking=ChunkingSettings(target_tokens=200, max_tokens=350, min_tokens=75, overlap_sentences=1),
        embedding=EmbeddingSettings(
            model="BAAI/bge-small-en-v1.5",
            query_prefix="Represent this sentence for searching relevant passages: ",
            batch_size=32,
        ),
    )


def helper_parts() -> tuple[SessionGuard, SignInCommand, SessionCommand]:
    settings = helper_settings()
    database = Database(settings.database_url)
    users = UserStore(database, SecretSealer(settings.data_key), BlindIndex(settings.data_key))
    sessions = SessionCommand(users, SessionStore(database))
    email_box = EmailBox(settings)
    sign_in = SignInCommand(
        settings,
        users,
        MagicLinkStore(database),
        sessions,
        EmailComposer(),
        email_box,
        JobQueue("mail", Alerter(email_box, "")),
    )
    return SessionGuard(settings, sessions), sign_in, sessions


def helper_instance() -> AuthRouter:
    return AuthRouter(*helper_parts())


def helper_request() -> Request:
    headers = [(b"origin", b"https://docs.example.com"), (b"cookie", b"rag_my_docs_session=tok-1")]
    return Request({"type": "http", "method": "POST", "path": "/api/auth/x", "headers": headers})


def test___init__() -> None:
    guard, sign_in, sessions = helper_parts()
    tested = AuthRouter(guard, sign_in, sessions)
    result = (tested._guard, tested._sign_in, tested._sessions)
    expected = (guard, sign_in, sessions)
    assert result == expected


def test_router() -> None:
    tested = helper_instance()
    result = [(route.path, tuple(sorted(route.methods)), route.name) for route in tested.router().routes]  # type: ignore[attr-defined]
    expected = [
        ("/api/auth/link-request", ("POST",), "link_requested"),
        ("/api/auth/link", ("POST",), "link_redeemed"),
        ("/api/auth/sign-out", ("POST",), "signed_out"),
    ]
    assert result == expected


def test_link_requested() -> None:
    tested = helper_instance()
    request = helper_request()
    with (
        patch.object(SessionGuard, "same_origin", side_effect=[None]) as mock_same_origin,
        patch.object(SignInCommand, "requested", side_effect=["On its way."]) as mock_requested,
    ):
        result = tested.link_requested(request, SignInRequest(email="ada@example.com"))
    expected = {"message": "On its way."}
    assert result == expected
    exp_calls = [call(request)]
    assert mock_same_origin.mock_calls == exp_calls
    exp_calls = [call("ada@example.com")]
    assert mock_requested.mock_calls == exp_calls


def test_link_redeemed() -> None:
    tested = helper_instance()
    request = helper_request()
    response = JSONResponse({"ok": True})
    with (
        patch.object(SessionGuard, "same_origin", side_effect=[None]) as mock_same_origin,
        patch.object(SignInCommand, "redeemed", side_effect=["session-tok"]) as mock_redeemed,
        patch.object(SessionGuard, "signed_in", side_effect=[response]) as mock_signed_in,
    ):
        result = tested.link_redeemed(request, TokenRequest(token="link-tok"))
    assert result is response
    exp_calls = [call(request)]
    assert mock_same_origin.mock_calls == exp_calls
    exp_calls = [call("link-tok")]
    assert mock_redeemed.mock_calls == exp_calls
    exp_calls = [call("session-tok", {"ok": True})]
    assert mock_signed_in.mock_calls == exp_calls


def test_signed_out() -> None:
    tested = helper_instance()
    request = helper_request()
    response = JSONResponse({"ok": True})
    with (
        patch.object(SessionGuard, "same_origin", side_effect=[None]) as mock_same_origin,
        patch.object(SessionCommand, "ended", side_effect=[None]) as mock_ended,
        patch.object(SessionGuard, "signed_out", side_effect=[response]) as mock_signed_out,
    ):
        result = tested.signed_out(request)
    assert result is response
    exp_calls = [call(request)]
    assert mock_same_origin.mock_calls == exp_calls
    exp_calls = [call("tok-1")]
    assert mock_ended.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_signed_out.mock_calls == exp_calls
