from __future__ import annotations

from unittest.mock import call, patch

import pytest

from rag_my_docs.commands.document_command import DocumentCommand
from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.document_store import DocumentStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.document import Document
from rag_my_docs.structures.document_file import DocumentFile
from rag_my_docs.structures.user_account import UserAccount

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="


def helper_instance() -> DocumentCommand:
    return DocumentCommand(DocumentStore(Database("postgresql://u:p@h/d"), SecretSealer(KEY)))


def helper_user() -> UserAccount:
    return UserAccount(user_id=7, email="owner@example.com", display_name="Owner", is_disabled=False)


def helper_document() -> Document:
    return Document(
        document_id=3,
        user_id=7,
        name="Publication 17",
        media_type=MediaType.PDF,
        size_bytes=9,
        created_at="2026-10-20T12:00:00Z",
        processing_status=ProcessingStatus.PENDING,
        index_status=IndexStatus.NOT_INDEXED,
        chunk_count=0,
        indexed_count=0,
    )


def test___init__() -> None:
    store = DocumentStore(Database("postgresql://u:p@h/d"), SecretSealer(KEY))
    tested = DocumentCommand(store)
    result = tested._documents
    expected = store
    assert result == expected


def test_documents() -> None:
    tested = helper_instance()
    with patch.object(DocumentStore, "of_user", side_effect=[[helper_document()]]) as mock_of_user:
        result = tested.documents(helper_user())
    expected = [helper_document()]
    assert result == expected
    exp_calls = [call(7)]
    assert mock_of_user.mock_calls == exp_calls


def test_downloaded() -> None:
    tested = helper_instance()
    file = DocumentFile(name="Publication 17", media_type=MediaType.PDF, content=b"%PDF-1.7\n")
    with patch.object(DocumentStore, "file_of", side_effect=[file]) as mock_file_of:
        result = tested.downloaded(helper_user(), 3)
    expected = DocumentFile(name="Publication 17", media_type=MediaType.PDF, content=b"%PDF-1.7\n")
    assert result == expected
    exp_calls = [call(7, 3)]
    assert mock_file_of.mock_calls == exp_calls


def test_downloaded__not_found() -> None:
    tested = helper_instance()
    with (
        patch.object(DocumentStore, "file_of", side_effect=[None]) as mock_file_of,
        pytest.raises(AppException) as exc_info,
    ):
        tested.downloaded(helper_user(), 99)
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (404, "This document does not exist.")
    assert result == expected
    exp_calls = [call(7, 99)]
    assert mock_file_of.mock_calls == exp_calls
