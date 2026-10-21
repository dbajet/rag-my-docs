from __future__ import annotations

from unittest.mock import call, patch

import pytest
from starlette.requests import Request

from rag_my_docs.commands.document_command import DocumentCommand
from rag_my_docs.commands.session_command import SessionCommand
from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.handlers.document_router import DocumentRouter
from rag_my_docs.handlers.session_guard import SessionGuard
from rag_my_docs.libraries.blind_index import BlindIndex
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.document_store import DocumentStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.session_store import SessionStore
from rag_my_docs.libraries.user_store import UserStore
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.document import Document
from rag_my_docs.structures.document_file import DocumentFile
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
        embedding=EmbeddingSettings(
            model="BAAI/bge-small-en-v1.5",
            query_prefix="Represent this sentence for searching relevant passages: ",
            batch_size=32,
        ),
    )


def helper_parts() -> tuple[SessionGuard, DocumentCommand]:
    settings = helper_settings()
    database = Database(settings.database_url)
    sealer = SecretSealer(settings.data_key)
    users = UserStore(database, sealer, BlindIndex(settings.data_key))
    guard = SessionGuard(settings, SessionCommand(users, SessionStore(database)))
    return guard, DocumentCommand(DocumentStore(database, sealer))


def helper_instance() -> DocumentRouter:
    return DocumentRouter(*helper_parts())


def helper_request() -> Request:
    return Request({"type": "http", "method": "GET", "path": "/api/documents", "headers": []})


def helper_user() -> UserAccount:
    return UserAccount(
        user_id=7,
        email="ada@example.com",
        display_name="Ada",
        is_disabled=False,
    )


def helper_document() -> Document:
    return Document(
        document_id=3,
        user_id=7,
        name="Résumé été.pdf",
        media_type=MediaType.PDF,
        size_bytes=9,
        created_at="2026-10-19T12:00:00Z",
        processing_status=ProcessingStatus.PENDING,
        index_status=IndexStatus.NOT_INDEXED,
        chunk_count=0,
        indexed_count=0,
    )


def test___init__() -> None:
    guard, documents = helper_parts()
    tested = DocumentRouter(guard, documents)
    result = (tested._guard, tested._documents)
    expected = (guard, documents)
    assert result == expected


def test_router() -> None:
    tested = helper_instance()
    result = [(route.path, tuple(sorted(route.methods)), route.name) for route in tested.router().routes]  # type: ignore[attr-defined]
    expected = [
        ("/api/documents", ("GET",), "documents"),
        ("/api/documents/{document_id}/content", ("GET",), "downloaded"),
    ]
    assert result == expected


def test_documents() -> None:
    tested = helper_instance()
    request = helper_request()
    with (
        patch.object(SessionGuard, "user", side_effect=[helper_user()]) as mock_user,
        patch.object(DocumentCommand, "documents", side_effect=[[helper_document()]]) as mock_documents,
    ):
        result = tested.documents(request)
    expected = {
        "documents": [
            {
                "document_id": 3,
                "user_id": 7,
                "name": "Résumé été.pdf",
                "media_type": "application/pdf",
                "size_bytes": 9,
                "created_at": "2026-10-19T12:00:00Z",
                "processing_status": "pending",
                "index_status": "not_indexed",
                "chunk_count": 0,
                "indexed_count": 0,
            }
        ]
    }
    assert result == expected
    exp_calls = [call(request)]
    assert mock_user.mock_calls == exp_calls
    exp_calls = [call(helper_user())]
    assert mock_documents.mock_calls == exp_calls


def test_downloaded() -> None:
    tested = helper_instance()
    request = helper_request()
    file = DocumentFile(name="notes.txt", media_type=MediaType.TEXT, content=b"hello\n")
    with (
        patch.object(SessionGuard, "user", side_effect=[helper_user()]) as mock_user,
        patch.object(DocumentCommand, "downloaded", side_effect=[file]) as mock_downloaded,
    ):
        response = tested.downloaded(request, 3)
    result = (response.status_code, response.body, dict(response.headers))
    expected = (
        200,
        b"hello\n",
        {
            "content-disposition": "attachment; filename=\"notes.txt\"; filename*=UTF-8''notes.txt",
            "cache-control": "private, no-store",
            "content-length": "6",
            "content-type": "text/plain; charset=utf-8",
        },
    )
    assert result == expected
    exp_calls = [call(request)]
    assert mock_user.mock_calls == exp_calls
    exp_calls = [call(helper_user(), 3)]
    assert mock_downloaded.mock_calls == exp_calls


@pytest.mark.parametrize(
    "name, expected",
    [
        pytest.param(
            "report 2026.pdf",
            "attachment; filename=\"report 2026.pdf\"; filename*=UTF-8''report%202026.pdf",
            id="ascii",
        ),
        pytest.param(
            'say "hi"\\now.txt',
            "attachment; filename=\"say _hi__now.txt\"; filename*=UTF-8''say%20%22hi%22%5Cnow.txt",
            id="quotes_and_backslash",
        ),
        pytest.param(
            "Résumé été.pdf",
            "attachment; filename=\"R_sum_ _t_.pdf\"; filename*=UTF-8''R%C3%A9sum%C3%A9%20%C3%A9t%C3%A9.pdf",
            id="non_ascii",
        ),
        pytest.param(
            "cat 🐈.png",
            "attachment; filename=\"cat _.png\"; filename*=UTF-8''cat%20%F0%9F%90%88.png",
            id="emoji",
        ),
    ],
)
def test_disposition(name: str, expected: str) -> None:
    tested = DocumentRouter
    result = tested.disposition(DocumentFile(name=name, media_type=MediaType.PDF, content=b""))
    assert result == expected
