from __future__ import annotations

import logging
from unittest.mock import call, patch

import pytest

from rag_my_docs.commands.corpus_command import CorpusCommand
from rag_my_docs.commands.index_command import IndexCommand
from rag_my_docs.commands.processing_command import ProcessingCommand
from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.libraries.alerter import Alerter
from rag_my_docs.libraries.chunker import Chunker
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.document_store import DocumentStore
from rag_my_docs.libraries.email_box import EmailBox
from rag_my_docs.libraries.embedding_box import EmbeddingBox
from rag_my_docs.libraries.index_store import IndexStore
from rag_my_docs.libraries.job_queue import JobQueue
from rag_my_docs.libraries.page_furniture import PageFurniture
from rag_my_docs.libraries.page_layout import PageLayout
from rag_my_docs.libraries.pdf_box import PdfBox
from rag_my_docs.libraries.processing_store import ProcessingStore
from rag_my_docs.libraries.publication_box import PublicationBox
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_detector import SectionDetector
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.libraries.sentence_splitter import SentenceSplitter
from rag_my_docs.libraries.text_digest import TextDigest
from rag_my_docs.libraries.token_counter import TokenCounter
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.document import Document
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.publication import Publication
from rag_my_docs.structures.settings import Settings
from rag_my_docs.structures.user_account import UserAccount

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="
# The SHA-256 of b"pdf bytes", computed with sha256sum.
PDF_SHA256 = "d1cb546b102fab8362de413fdacc187b05be10df72b72db3b3e50b4953f6a555"


def helper_settings() -> Settings:
    return Settings(
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


def helper_parts() -> tuple[DocumentStore, ProcessingCommand, JobQueue]:
    settings = helper_settings()
    database = Database(settings.database_url)
    sealer = SecretSealer(KEY)
    alerter = Alerter(EmailBox(settings), "")
    indexing = IndexCommand(
        IndexStore(database, sealer, SectionPaths(database, sealer)),
        EmbeddingBox(settings.embedding),
        JobQueue("indexing", alerter),
    )
    processing = ProcessingCommand(
        ProcessingStore(database, sealer, TextDigest(KEY)),
        PdfBox(),
        PageLayout(),
        PageFurniture(),
        SectionDetector(),
        Chunker(settings.chunking, SentenceSplitter(), TokenCounter()),
        settings.chunking.strategy(),
        JobQueue("processing", alerter),
        indexing,
    )
    return DocumentStore(database, sealer), processing, JobQueue("corpus", alerter)


def helper_publications() -> list[Publication]:
    return [
        Publication("Publication 17", "https://www.irs.gov/pub/irs-pdf/p17.pdf", PDF_SHA256),
        Publication("Publication 501", "https://www.irs.gov/pub/irs-pdf/p501.pdf", PDF_SHA256),
    ]


def helper_instance() -> CorpusCommand:
    documents, processing, jobs = helper_parts()
    return CorpusCommand(helper_publications(), PublicationBox(), documents, processing, jobs)


def helper_owner() -> UserAccount:
    return UserAccount(user_id=7, email="owner@example.com", display_name="owner", is_disabled=False)


def helper_document(document_id: int, name: str) -> Document:
    return Document(
        document_id=document_id,
        user_id=7,
        name=name,
        media_type=MediaType.PDF,
        size_bytes=9,
        created_at="2026-10-20T12:00:00Z",
        processing_status=ProcessingStatus.PENDING,
        index_status=IndexStatus.NOT_INDEXED,
        chunk_count=0,
        indexed_count=0,
    )


def test___init__() -> None:
    documents, processing, jobs = helper_parts()
    publications = helper_publications()
    box = PublicationBox()
    tested = CorpusCommand(publications, box, documents, processing, jobs)
    result = (tested._publications, tested._box, tested._documents, tested._processing, tested._jobs)
    expected = (publications, box, documents, processing, jobs)
    assert result == expected


def test_queued() -> None:
    tested = helper_instance()
    with patch.object(JobQueue, "queued", side_effect=[None]) as mock_queued:
        result = tested.queued(helper_owner())
    assert result is None
    exp_calls = [call(tested.synced, (helper_owner(),))]
    assert mock_queued.mock_calls == exp_calls


def test_synced() -> None:
    """A stored document the list does not name is deleted; a publication
    already stored is not downloaded again."""
    tested = helper_instance()
    stored = [helper_document(3, "Trans Tahoe 2016.pdf"), helper_document(4, "Publication 501")]
    with (
        patch.object(DocumentStore, "of_user", side_effect=[stored]) as mock_of_user,
        patch.object(DocumentStore, "deleted", side_effect=[1]) as mock_deleted,
        patch.object(CorpusCommand, "loaded", side_effect=[None]) as mock_loaded,
    ):
        result = tested.synced(helper_owner())
    assert result is None
    exp_calls = [call(7)]
    assert mock_of_user.mock_calls == exp_calls
    exp_calls = [call(7, 3)]
    assert mock_deleted.mock_calls == exp_calls
    exp_calls = [call(helper_owner(), helper_publications()[0])]
    assert mock_loaded.mock_calls == exp_calls


def test_loaded() -> None:
    tested = helper_instance()
    with (
        patch.object(PublicationBox, "downloaded", side_effect=[b"pdf bytes"]) as mock_downloaded,
        patch.object(DocumentStore, "created", side_effect=[helper_document(9, "Publication 17")]) as mock_created,
        patch.object(ProcessingCommand, "queued", side_effect=[None]) as mock_queued,
    ):
        result = tested.loaded(helper_owner(), helper_publications()[0])
    assert result is None
    exp_calls = [call("https://www.irs.gov/pub/irs-pdf/p17.pdf")]
    assert mock_downloaded.mock_calls == exp_calls
    exp_calls = [call(7, "Publication 17", MediaType.PDF, b"pdf bytes", ProcessingStatus.PENDING)]
    assert mock_created.mock_calls == exp_calls
    exp_calls = [call(9)]
    assert mock_queued.mock_calls == exp_calls


@pytest.mark.parametrize(
    "outcome, exp_message",
    [
        pytest.param(
            AppException(502, "The publication could not be downloaded."),
            "[CORPUS] https://www.irs.gov/pub/irs-pdf/p17.pdf not loaded: The publication could not be downloaded.",
            id="download-failed",
        ),
        pytest.param(
            b"revised bytes",
            "[CORPUS] https://www.irs.gov/pub/irs-pdf/p17.pdf not loaded: its checksum changed, 13 bytes",
            id="checksum-changed",
        ),
    ],
)
def test_loaded__not_stored(caplog: pytest.LogCaptureFixture, outcome: object, exp_message: str) -> None:
    """Logged and left for the next start; nothing is stored or queued."""
    tested = helper_instance()
    with (
        caplog.at_level(logging.WARNING, logger="rag_my_docs"),
        patch.object(PublicationBox, "downloaded", side_effect=[outcome]) as mock_downloaded,
        patch.object(DocumentStore, "created", side_effect=[]) as mock_created,
        patch.object(ProcessingCommand, "queued", side_effect=[]) as mock_queued,
    ):
        result = tested.loaded(helper_owner(), helper_publications()[0])
    assert result is None
    assert caplog.messages == [exp_message]
    exp_calls = [call("https://www.irs.gov/pub/irs-pdf/p17.pdf")]
    assert mock_downloaded.mock_calls == exp_calls
    exp_calls = []
    assert mock_created.mock_calls == exp_calls
    assert mock_queued.mock_calls == exp_calls
