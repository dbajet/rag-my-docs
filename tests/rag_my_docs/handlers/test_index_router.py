from __future__ import annotations

from unittest.mock import call, patch

from starlette.requests import Request

from rag_my_docs.commands.index_command import IndexCommand
from rag_my_docs.commands.session_command import SessionCommand
from rag_my_docs.handlers.index_request import IndexRequest
from rag_my_docs.handlers.index_router import IndexRouter
from rag_my_docs.handlers.session_guard import SessionGuard
from rag_my_docs.libraries.alerter import Alerter
from rag_my_docs.libraries.blind_index import BlindIndex
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.email_box import EmailBox
from rag_my_docs.libraries.embedding_box import EmbeddingBox
from rag_my_docs.libraries.index_store import IndexStore
from rag_my_docs.libraries.job_queue import JobQueue
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.libraries.session_store import SessionStore
from rag_my_docs.libraries.user_store import UserStore
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.settings import Settings
from rag_my_docs.structures.user_account import UserAccount


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
        embedding=EmbeddingSettings(model="BAAI/bge-small-en-v1.5", query_prefix="Query: ", batch_size=32),
    )


def helper_parts() -> tuple[SessionGuard, IndexCommand]:
    settings = helper_settings()
    database = Database(settings.database_url)
    sealer = SecretSealer(settings.data_key)
    users = UserStore(database, sealer, BlindIndex(settings.data_key))
    guard = SessionGuard(settings, SessionCommand(users, SessionStore(database)))
    indexing = IndexCommand(
        IndexStore(database, sealer, SectionPaths(database, sealer)),
        EmbeddingBox(settings.embedding),
        JobQueue("indexing", Alerter(EmailBox(settings), "")),
    )
    return guard, indexing


def helper_instance() -> IndexRouter:
    return IndexRouter(*helper_parts())


def helper_request(path: str) -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def helper_user() -> UserAccount:
    return UserAccount(
        user_id=7,
        email="ada@example.com",
        display_name="Ada",
        is_disabled=False,
    )


def test___init__() -> None:
    guard, indexing = helper_parts()
    tested = IndexRouter(guard, indexing)
    result = (tested._guard, tested._indexing)
    expected = (guard, indexing)
    assert result == expected


def test_router() -> None:
    tested = helper_instance()
    result = [(route.path, tuple(sorted(route.methods)), route.name) for route in tested.router().routes]  # type: ignore[attr-defined]
    expected = [
        ("/api/documents/{document_id}/index", ("POST",), "reindexed"),
        ("/api/index/all", ("POST",), "indexed_all"),
    ]
    assert result == expected


def test_reindexed() -> None:
    tested = helper_instance()
    request = helper_request("/api/documents/3/index")
    with (
        patch.object(SessionGuard, "user", side_effect=[helper_user()]) as mock_user,
        patch.object(IndexCommand, "reindexed", side_effect=[None]) as mock_reindexed,
    ):
        result = tested.reindexed(request, 3, IndexRequest(rebuild=True))
    expected = {"ok": True}
    assert result == expected
    exp_calls = [call(request)]
    assert mock_user.mock_calls == exp_calls
    exp_calls = [call(helper_user(), 3, True)]
    assert mock_reindexed.mock_calls == exp_calls


def test_indexed_all() -> None:
    tested = helper_instance()
    request = helper_request("/api/index/all")
    with (
        patch.object(SessionGuard, "user", side_effect=[helper_user()]) as mock_user,
        patch.object(IndexCommand, "indexed_all", side_effect=[4]) as mock_indexed_all,
    ):
        result = tested.indexed_all(request)
    expected = {"queued": 4}
    assert result == expected
    exp_calls = [call(request)]
    assert mock_user.mock_calls == exp_calls
    exp_calls = [call(helper_user())]
    assert mock_indexed_all.mock_calls == exp_calls
