from __future__ import annotations

from unittest.mock import call, patch

import pytest
from starlette.requests import Request

from rag_my_docs.commands.index_command import IndexCommand
from rag_my_docs.commands.processing_command import ProcessingCommand
from rag_my_docs.commands.session_command import SessionCommand
from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.handlers.processing_router import ProcessingRouter
from rag_my_docs.handlers.session_guard import SessionGuard
from rag_my_docs.libraries.alerter import Alerter
from rag_my_docs.libraries.blind_index import BlindIndex
from rag_my_docs.libraries.chunker import Chunker
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.email_box import EmailBox
from rag_my_docs.libraries.embedding_box import EmbeddingBox
from rag_my_docs.libraries.index_store import IndexStore
from rag_my_docs.libraries.job_queue import JobQueue
from rag_my_docs.libraries.page_furniture import PageFurniture
from rag_my_docs.libraries.page_layout import PageLayout
from rag_my_docs.libraries.pdf_box import PdfBox
from rag_my_docs.libraries.processing_store import ProcessingStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_detector import SectionDetector
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.libraries.sentence_splitter import SentenceSplitter
from rag_my_docs.libraries.session_store import SessionStore
from rag_my_docs.libraries.text_digest import TextDigest
from rag_my_docs.libraries.token_counter import TokenCounter
from rag_my_docs.libraries.user_store import UserStore
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.document_processing import DocumentProcessing
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.settings import Settings
from rag_my_docs.structures.source_span import SourceSpan
from rag_my_docs.structures.stored_chunk import StoredChunk
from rag_my_docs.structures.stored_section import StoredSection
from rag_my_docs.structures.text_block import TextBlock
from rag_my_docs.structures.user_account import UserAccount

STRATEGY = "sentences-v2 target=200 max=350 min=75 overlap=1"


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


def helper_parts() -> tuple[SessionGuard, ProcessingCommand]:
    settings = helper_settings()
    database = Database(settings.database_url)
    sealer = SecretSealer(settings.data_key)
    users = UserStore(database, sealer, BlindIndex(settings.data_key))
    guard = SessionGuard(settings, SessionCommand(users, SessionStore(database)))
    alerter = Alerter(EmailBox(settings), "")
    indexing = IndexCommand(
        IndexStore(database, sealer, SectionPaths(database, sealer)),
        EmbeddingBox(settings.embedding),
        JobQueue("indexing", alerter),
    )
    processing = ProcessingCommand(
        ProcessingStore(database, sealer, TextDigest(settings.data_key)),
        PdfBox(),
        PageLayout(),
        PageFurniture(),
        SectionDetector(),
        Chunker(settings.chunking, SentenceSplitter(), TokenCounter()),
        settings.chunking.strategy(),
        JobQueue("processing", alerter),
        indexing,
    )
    return guard, processing


def helper_instance() -> ProcessingRouter:
    return ProcessingRouter(*helper_parts())


def helper_request(method: str) -> Request:
    return Request({"type": "http", "method": method, "path": "/api/documents/3/processing", "headers": []})


def helper_user() -> UserAccount:
    return UserAccount(
        user_id=7,
        email="ada@example.com",
        display_name="Ada",
        is_disabled=False,
    )


def helper_processing(chunking_version: str) -> DocumentProcessing:
    return DocumentProcessing(
        document_id=3,
        name="report.pdf",
        media_type=MediaType.PDF,
        status=ProcessingStatus.COMPLETED,
        error="",
        page_count=2,
        chunking_version=chunking_version,
        processed_at="2026-10-19T12:00:00Z",
        report={"tables": 0},
        index_status=IndexStatus.INDEXED,
        index_model="BAAI/bge-small-en-v1.5",
        index_error="",
        indexed_at="2026-10-19T12:01:00Z",
        index_report={"chunks": 1, "embedded": 1, "reused": 0, "batches": 1, "duration_ms": 12.5},
        chunk_count=1,
        indexed_count=1,
    )


def helper_section() -> StoredSection:
    return StoredSection(
        section_id=11,
        parent_id=None,
        position=0,
        level=1,
        number="1",
        title="1. Intro",
        source=SectionSource.NUMBERED,
        start_page=1,
        end_page=2,
        heading_block_id=0,
        chunk_count=1,
    )


def helper_chunk() -> StoredChunk:
    return StoredChunk(
        chunk_id=21,
        section_id=11,
        position=0,
        kind=ChunkKind.PROSE,
        text="One sentence.",
        prefix_chars=0,
        token_count=3,
        start_page=1,
        end_page=1,
        spans=(SourceSpan(block_id=1, start=0, end=13),),
        strategy=STRATEGY,
        embedding_model="BAAI/bge-small-en-v1.5",
    )


def test___init__() -> None:
    guard, processing = helper_parts()
    tested = ProcessingRouter(guard, processing)
    result = (tested._guard, tested._processing)
    expected = (guard, processing)
    assert result == expected


def test_router() -> None:
    tested = helper_instance()
    result = [(route.path, tuple(sorted(route.methods)), route.name) for route in tested.router().routes]  # type: ignore[attr-defined]
    expected = [
        ("/api/documents/{document_id}/processing", ("GET",), "processing"),
        ("/api/documents/{document_id}/processing", ("POST",), "reprocessed"),
        ("/api/documents/{document_id}/extraction", ("GET",), "extraction"),
    ]
    assert result == expected


@pytest.mark.parametrize(
    "chunking_version, exp_stale",
    [
        pytest.param(STRATEGY, False, id="current_settings"),
        pytest.param("sentences-v2 target=100 max=150 min=30 overlap=0", True, id="other_settings"),
        pytest.param("", False, id="never_processed"),
    ],
)
def test_processing(chunking_version: str, exp_stale: bool) -> None:
    tested = helper_instance()
    request = helper_request("GET")
    with (
        patch.object(SessionGuard, "user", side_effect=[helper_user()]) as mock_user,
        patch.object(
            ProcessingCommand,
            "processing",
            side_effect=[(helper_processing(chunking_version), [helper_section()], [helper_chunk()])],
        ) as mock_processing,
    ):
        result = tested.processing(request, 3)
    expected = {
        "document": {
            "document_id": 3,
            "name": "report.pdf",
            "media_type": "application/pdf",
            "status": "completed",
            "error": "",
            "page_count": 2,
            "chunking_version": chunking_version,
            "processed_at": "2026-10-19T12:00:00Z",
            "report": {"tables": 0},
            "index_status": "indexed",
            "index_model": "BAAI/bge-small-en-v1.5",
            "index_error": "",
            "indexed_at": "2026-10-19T12:01:00Z",
            "index_report": {"chunks": 1, "embedded": 1, "reused": 0, "batches": 1, "duration_ms": 12.5},
            "chunk_count": 1,
            "indexed_count": 1,
            "current_strategy": STRATEGY,
            "is_stale": exp_stale,
        },
        "sections": [
            {
                "section_id": 11,
                "parent_id": None,
                "position": 0,
                "level": 1,
                "number": "1",
                "title": "1. Intro",
                "source": "numbered",
                "start_page": 1,
                "end_page": 2,
                "heading_block_id": 0,
                "chunk_count": 1,
            }
        ],
        "chunks": [
            {
                "chunk_id": 21,
                "section_id": 11,
                "position": 0,
                "kind": "prose",
                "text": "One sentence.",
                "prefix_chars": 0,
                "token_count": 3,
                "start_page": 1,
                "end_page": 1,
                "spans": [{"block_id": 1, "start": 0, "end": 13}],
                "strategy": STRATEGY,
                "embedding_model": "BAAI/bge-small-en-v1.5",
            }
        ],
    }
    assert result == expected
    exp_calls = [call(request)]
    assert mock_user.mock_calls == exp_calls
    exp_calls = [call(helper_user(), 3)]
    assert mock_processing.mock_calls == exp_calls


def test_reprocessed() -> None:
    tested = helper_instance()
    request = helper_request("POST")
    with (
        patch.object(SessionGuard, "user", side_effect=[helper_user()]) as mock_user,
        patch.object(ProcessingCommand, "reprocessed", side_effect=[None]) as mock_reprocessed,
    ):
        result = tested.reprocessed(request, 3)
    expected = {"ok": True}
    assert result == expected
    exp_calls = [call(request)]
    assert mock_user.mock_calls == exp_calls
    exp_calls = [call(helper_user(), 3)]
    assert mock_reprocessed.mock_calls == exp_calls


def test_extraction() -> None:
    tested = helper_instance()
    request = helper_request("GET")
    block = TextBlock(
        block_id=0,
        page=1,
        kind=BlockKind.HEADER,
        text="ACME report",
        font_size=8.0,
        bold_share=0.0,
        line_count=1,
        top=0.04,
        bottom=0.06,
    )
    with (
        patch.object(SessionGuard, "user", side_effect=[helper_user()]) as mock_user,
        patch.object(ProcessingCommand, "extraction", side_effect=[[block]]) as mock_extraction,
    ):
        result = tested.extraction(request, 3)
    expected = {
        "blocks": [
            {
                "block_id": 0,
                "page": 1,
                "kind": "header",
                "text": "ACME report",
                "font_size": 8.0,
                "bold_share": 0.0,
                "line_count": 1,
                "top": 0.04,
                "bottom": 0.06,
            }
        ]
    }
    assert result == expected
    exp_calls = [call(request)]
    assert mock_user.mock_calls == exp_calls
    exp_calls = [call(helper_user(), 3)]
    assert mock_extraction.mock_calls == exp_calls
