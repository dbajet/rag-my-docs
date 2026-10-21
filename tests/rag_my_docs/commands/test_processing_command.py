from __future__ import annotations

import logging
from typing import Any
from unittest.mock import call, patch

import pytest

from rag_my_docs.commands.index_command import IndexCommand
from rag_my_docs.commands.processing_command import ProcessingCommand
from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.libraries.alerter import Alerter
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
from rag_my_docs.libraries.text_digest import TextDigest
from rag_my_docs.libraries.token_counter import TokenCounter
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.chunk import Chunk
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.claimed_document import ClaimedDocument
from rag_my_docs.structures.document_processing import DocumentProcessing
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.outline_entry import OutlineEntry
from rag_my_docs.structures.pdf_document import PdfDocument
from rag_my_docs.structures.pdf_page import PdfPage
from rag_my_docs.structures.processing_result import ProcessingResult
from rag_my_docs.structures.raw_block import RawBlock
from rag_my_docs.structures.section import Section
from rag_my_docs.structures.settings import Settings
from rag_my_docs.structures.source_span import SourceSpan
from rag_my_docs.structures.stored_chunk import StoredChunk
from rag_my_docs.structures.stored_section import StoredSection
from rag_my_docs.structures.text_block import TextBlock
from rag_my_docs.structures.user_account import UserAccount
from tests.support.pdf_factory import PdfFactory

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="


def helper_chunking(target: int, maximum: int, minimum: int, overlap: int) -> ChunkingSettings:
    return ChunkingSettings(target_tokens=target, max_tokens=maximum, min_tokens=minimum, overlap_sentences=overlap)


def helper_settings() -> Settings:
    return Settings(
        environment="production",
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
        email_from="noreply@example.com",
        bootstrap_admin_email="",
        chunking=helper_chunking(200, 350, 75, 1),
        embedding=EmbeddingSettings(model="BAAI/bge-small-en-v1.5", query_prefix="Query: ", batch_size=32),
    )


def helper_store() -> ProcessingStore:
    return ProcessingStore(Database("postgresql://u:p@h/d"), SecretSealer(KEY), TextDigest(KEY))


def helper_indexing() -> IndexCommand:
    settings = helper_settings()
    return IndexCommand(
        IndexStore(
            Database("postgresql://u:p@h/d"),
            SecretSealer(KEY),
            SectionPaths(Database("postgresql://u:p@h/d"), SecretSealer(KEY)),
        ),
        EmbeddingBox(settings.embedding),
        JobQueue("indexing", Alerter(EmailBox(settings), "")),
    )


def helper_instance(chunking: ChunkingSettings) -> ProcessingCommand:
    return ProcessingCommand(
        helper_store(),
        PdfBox(),
        PageLayout(),
        PageFurniture(),
        SectionDetector(),
        Chunker(chunking, SentenceSplitter(), TokenCounter()),
        chunking.strategy(),
        JobQueue("processing", Alerter(EmailBox(helper_settings()), "")),
        helper_indexing(),
    )


def helper_user() -> UserAccount:
    return UserAccount(
        user_id=7,
        email="ada@example.com",
        display_name="Ada",
        is_disabled=False,
    )


def helper_claimed(media_type: MediaType) -> ClaimedDocument:
    return ClaimedDocument(document_id=5, run=2, media_type=media_type, content=b"%PDF-1.7 bytes")


def helper_result(chunks: tuple[Chunk, ...], sections: tuple[Section, ...]) -> ProcessingResult:
    return ProcessingResult(page_count=3, blocks=(), sections=sections, chunks=chunks, report={"blocks": 0})


def helper_section(heading_block_id: int | None) -> Section:
    return Section(
        position=0,
        parent_position=None,
        level=1,
        number="",
        title="",
        source=SectionSource.FALLBACK,
        heading_block_id=heading_block_id,
        block_ids=(),
        start_page=1,
        end_page=1,
    )


def helper_chunk() -> Chunk:
    return Chunk(
        position=0,
        section_position=0,
        kind=ChunkKind.PROSE,
        text="Some text.",
        prefix_chars=0,
        token_count=3,
        start_page=1,
        end_page=1,
        spans=(SourceSpan(block_id=0, start=0, end=10),),
    )


def helper_processing(media_type: MediaType) -> DocumentProcessing:
    return DocumentProcessing(
        document_id=5,
        name="report.pdf",
        media_type=media_type,
        status=ProcessingStatus.COMPLETED,
        error="",
        page_count=3,
        chunking_version="sentences-v2 target=200 max=350 min=75 overlap=1",
        processed_at="2026-10-19T10:00:00Z",
        report={},
        index_status=IndexStatus.INDEXED,
        index_model="BAAI/bge-small-en-v1.5",
        index_error="",
        indexed_at="2026-10-19T10:01:00Z",
        index_report={},
        chunk_count=4,
        indexed_count=4,
    )


def helper_furnished(factory: PdfFactory, page: int) -> PdfFactory:
    """A page with a running header and a page-number footer, as most reports have."""
    return (
        factory.page()
        .text_in((72, 30, 523, 50), "ACME Annual Report", 8, False)
        .text_in((72, 800, 523, 820), f"Page {page}", 8, False)
    )


def helper_report_pdf() -> bytes:
    """Three pages: numbered nested headings, running headers and footers, a
    paragraph broken by a page break, two columns, and a ruled table."""
    factory = PdfFactory()
    helper_furnished(factory, 1)
    factory.paragraph("1. Introduction", 14, True)
    factory.paragraph("This report covers the year. Costs rose by 3.5 percent in total.", 10, False)
    factory.paragraph("1.1 Background", 12, True)
    factory.paragraph("The committee met twice. Its findings continue on the", 10, False)
    helper_furnished(factory, 2)
    factory.paragraph("next page without a break. A new sentence ends it.", 10, False)
    factory.paragraph("2. Results", 14, True)
    factory.text_in((72, 160, 280, 230), "Left column first. Left column second.", 10, False)
    factory.text_in((72, 240, 280, 300), "Left column third.", 10, False)
    factory.text_in((315, 160, 523, 230), "Right column first. Right column second.", 10, False)
    factory.text_in((315, 240, 523, 300), "Right column third.", 10, False)
    factory.table([["Region", "Sales"], ["North", "120"], ["South", "95"]], 330)
    helper_furnished(factory, 3)
    factory.paragraph("2.1 Details", 12, True)
    factory.paragraph("Details close the report.", 10, False)
    return factory.built()


def helper_shapes(result: ProcessingResult) -> tuple[Any, ...]:
    """The result as plain tuples: blocks, sections, chunks, in that order."""
    return (
        result.page_count,
        [(block.block_id, block.page, block.kind.value, block.text) for block in result.blocks],
        [
            (
                section.position,
                section.parent_position,
                section.level,
                section.number,
                section.title,
                section.source.value,
                section.heading_block_id,
                section.block_ids,
                section.start_page,
                section.end_page,
            )
            for section in result.sections
        ],
        [
            (
                chunk.position,
                chunk.section_position,
                chunk.kind.value,
                chunk.text,
                chunk.prefix_chars,
                chunk.token_count,
                chunk.start_page,
                chunk.end_page,
                tuple(tuple(span) for span in chunk.spans),
            )
            for chunk in result.chunks
        ],
        result.report,
    )


def test___init__() -> None:
    store = helper_store()
    pdf_box = PdfBox()
    layout = PageLayout()
    furniture = PageFurniture()
    detector = SectionDetector()
    chunker = Chunker(helper_chunking(200, 350, 75, 1), SentenceSplitter(), TokenCounter())
    jobs = JobQueue("processing", Alerter(EmailBox(helper_settings()), ""))
    indexing = helper_indexing()
    tested = ProcessingCommand(store, pdf_box, layout, furniture, detector, chunker, "strategy-x", jobs, indexing)
    result = (
        tested._store,
        tested._pdf_box,
        tested._layout,
        tested._furniture,
        tested._detector,
        tested._chunker,
        tested._strategy,
        tested._jobs,
        tested._indexing,
    )
    expected = (store, pdf_box, layout, furniture, detector, chunker, "strategy-x", jobs, indexing)
    assert result == expected


def test_queued() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    with patch.object(JobQueue, "queued", side_effect=[None]) as mock_queued:
        result = tested.queued(5)
    assert result is None
    exp_calls = [call(tested.processed, (5,))]
    assert mock_queued.mock_calls == exp_calls


def test_resumed() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    with (
        patch.object(ProcessingStore, "restarted", side_effect=[1]) as mock_restarted,
        patch.object(ProcessingStore, "pending_ids", side_effect=[[3, 4]]) as mock_pending_ids,
        patch.object(ProcessingCommand, "queued", side_effect=[None, None]) as mock_queued,
    ):
        result = tested.resumed()
    assert result is None
    exp_calls = [call(10)]
    assert mock_restarted.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_pending_ids.mock_calls == exp_calls
    exp_calls = [call(3), call(4)]
    assert mock_queued.mock_calls == exp_calls


def test_processed() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    claimed = helper_claimed(MediaType.PDF)
    processed = helper_result((helper_chunk(),), (helper_section(None),))
    with (
        patch.object(ProcessingStore, "claimed", side_effect=[claimed]) as mock_claimed,
        patch.object(ProcessingCommand, "result", side_effect=[processed]) as mock_result,
        patch.object(ProcessingCommand, "stored", side_effect=[None]) as mock_stored,
        patch.object(ProcessingStore, "closed", side_effect=[]) as mock_closed,
    ):
        result = tested.processed(5)
    assert result is None
    exp_calls = [call(5)]
    assert mock_claimed.mock_calls == exp_calls
    exp_calls = [call(b"%PDF-1.7 bytes")]
    assert mock_result.mock_calls == exp_calls
    exp_calls = [call(helper_claimed(MediaType.PDF), helper_result((helper_chunk(),), (helper_section(None),)))]
    assert mock_stored.mock_calls == exp_calls
    exp_calls = []
    assert mock_closed.mock_calls == exp_calls


def test_processed__not_waiting() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    with (
        patch.object(ProcessingStore, "claimed", side_effect=[None]) as mock_claimed,
        patch.object(ProcessingCommand, "result", side_effect=[]) as mock_result,
        patch.object(ProcessingStore, "closed", side_effect=[]) as mock_closed,
    ):
        result = tested.processed(5)
    assert result is None
    exp_calls = [call(5)]
    assert mock_claimed.mock_calls == exp_calls
    exp_calls = []
    assert mock_result.mock_calls == exp_calls
    assert mock_closed.mock_calls == exp_calls


def test_processed__not_a_pdf() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    with (
        patch.object(ProcessingStore, "claimed", side_effect=[helper_claimed(MediaType.PNG)]) as mock_claimed,
        patch.object(ProcessingCommand, "result", side_effect=[]) as mock_result,
        patch.object(ProcessingStore, "closed", side_effect=[True]) as mock_closed,
    ):
        result = tested.processed(5)
    assert result is None
    exp_calls = [call(5)]
    assert mock_claimed.mock_calls == exp_calls
    exp_calls = []
    assert mock_result.mock_calls == exp_calls
    exp_calls = [
        call(
            helper_claimed(MediaType.PNG),
            ProcessingStatus.UNSUPPORTED,
            "Only PDF files are processed for search for now.",
            {},
        )
    ]
    assert mock_closed.mock_calls == exp_calls


def test_processed__unreadable() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    with (
        patch.object(ProcessingStore, "claimed", side_effect=[helper_claimed(MediaType.PDF)]) as mock_claimed,
        patch.object(ProcessingStore, "closed", side_effect=[True]) as mock_closed,
        patch.object(ProcessingCommand, "stored", side_effect=[]) as mock_stored,
    ):
        result = tested.processed(5)
    assert result is None
    exp_calls = [call(5)]
    assert mock_claimed.mock_calls == exp_calls
    exp_calls = [
        call(
            helper_claimed(MediaType.PDF),
            ProcessingStatus.FAILED,
            "This PDF could not be read: the file is damaged or is not a PDF.",
            {},
        )
    ]
    assert mock_closed.mock_calls == exp_calls
    exp_calls = []
    assert mock_stored.mock_calls == exp_calls


def test_processed__unexpected_failure(caplog: pytest.LogCaptureFixture) -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    with (
        patch.object(ProcessingStore, "claimed", side_effect=[helper_claimed(MediaType.PDF)]) as mock_claimed,
        patch.object(ProcessingCommand, "result", side_effect=[ValueError("boom")]) as mock_result,
        patch.object(ProcessingStore, "closed", side_effect=[True]) as mock_closed,
        pytest.raises(ValueError) as exc_info,
    ):
        tested.processed(5)
    assert str(exc_info.value) == "boom"
    exp_calls = [call(5)]
    assert mock_claimed.mock_calls == exp_calls
    exp_calls = [call(b"%PDF-1.7 bytes")]
    assert mock_result.mock_calls == exp_calls
    exp_calls = [
        call(
            helper_claimed(MediaType.PDF),
            ProcessingStatus.FAILED,
            "Processing failed unexpectedly. Try again, or report it.",
            {},
        )
    ]
    assert mock_closed.mock_calls == exp_calls
    exp_messages = ["[PROCESSING] document 5 failed"]
    assert [record.getMessage() for record in caplog.records] == exp_messages


def test_stored(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="rag_my_docs")
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    processed = helper_result((helper_chunk(),), (helper_section(None),))
    with (
        patch.object(ProcessingStore, "finished", side_effect=[True]) as mock_finished,
        patch.object(ProcessingStore, "closed", side_effect=[]) as mock_closed,
        patch.object(IndexCommand, "queued", side_effect=[None]) as mock_queued,
    ):
        result = tested.stored(helper_claimed(MediaType.PDF), processed)
    assert result is None
    exp_calls = [
        call(
            helper_claimed(MediaType.PDF),
            helper_result((helper_chunk(),), (helper_section(None),)),
            "sentences-v2 target=200 max=350 min=75 overlap=1",
        )
    ]
    assert mock_finished.mock_calls == exp_calls
    exp_calls = []
    assert mock_closed.mock_calls == exp_calls
    # The next stage: the new chunks are queued for embedding.
    exp_calls = [call(5)]
    assert mock_queued.mock_calls == exp_calls
    exp_messages = ["[PROCESSING] document 5: 3 pages, 1 sections, 1 chunks"]
    assert [record.getMessage() for record in caplog.records] == exp_messages


def test_stored__headings_only_is_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="rag_my_docs")
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    processed = helper_result((), (helper_section(4),))
    with (
        patch.object(ProcessingStore, "finished", side_effect=[False]) as mock_finished,
        patch.object(ProcessingStore, "closed", side_effect=[]) as mock_closed,
        patch.object(IndexCommand, "queued", side_effect=[]) as mock_queued,
    ):
        result = tested.stored(helper_claimed(MediaType.PDF), processed)
    assert result is None
    exp_calls = [
        call(
            helper_claimed(MediaType.PDF),
            helper_result((), (helper_section(4),)),
            "sentences-v2 target=200 max=350 min=75 overlap=1",
        )
    ]
    assert mock_finished.mock_calls == exp_calls
    exp_calls = []
    assert mock_closed.mock_calls == exp_calls
    # Not stored: nothing to index.
    assert mock_queued.mock_calls == exp_calls
    # The run lost the document to a newer one: nothing to log.
    exp_messages: list[str] = []
    assert [record.getMessage() for record in caplog.records] == exp_messages


def test_stored__no_text() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    processed = helper_result((), ())
    with (
        patch.object(ProcessingStore, "finished", side_effect=[]) as mock_finished,
        patch.object(ProcessingStore, "closed", side_effect=[True]) as mock_closed,
        patch.object(IndexCommand, "queued", side_effect=[]) as mock_queued,
    ):
        result = tested.stored(helper_claimed(MediaType.PDF), processed)
    assert result is None
    exp_calls = []
    assert mock_finished.mock_calls == exp_calls
    exp_calls = [
        call(
            helper_claimed(MediaType.PDF),
            ProcessingStatus.NO_TEXT,
            "No page of this PDF has extractable text; it is probably scanned (OCR is not supported yet).",
            {"blocks": 0},
        )
    ]
    assert mock_closed.mock_calls == exp_calls
    exp_calls = []
    assert mock_queued.mock_calls == exp_calls


def test_result() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    result = helper_shapes(tested.result(helper_report_pdf()))
    expected = (
        3,
        [
            (0, 1, "header", "ACME Annual Report"),
            (1, 1, "heading", "1. Introduction"),
            (2, 1, "paragraph", "This report covers the year. Costs rose by 3.5 percent in total."),
            (3, 1, "heading", "1.1 Background"),
            (4, 1, "paragraph", "The committee met twice. Its findings continue on the"),
            (5, 1, "footer", "Page 1"),
            (6, 2, "header", "ACME Annual Report"),
            (7, 2, "paragraph", "next page without a break. A new sentence ends it."),
            (8, 2, "heading", "2. Results"),
            (9, 2, "paragraph", "Left column first. Left column second."),
            (10, 2, "paragraph", "Left column third."),
            (11, 2, "paragraph", "Right column first. Right column second."),
            (12, 2, "paragraph", "Right column third."),
            (13, 2, "table", "| Region | Sales |\n| North | 120 |\n| South | 95 |"),
            (14, 2, "footer", "Page 2"),
            (15, 3, "header", "ACME Annual Report"),
            (16, 3, "heading", "2.1 Details"),
            (17, 3, "paragraph", "Details close the report."),
            (18, 3, "footer", "Page 3"),
        ],
        [
            (0, None, 1, "1", "1. Introduction", "numbered", 1, (2,), 1, 2),
            (1, 0, 2, "1.1", "1.1 Background", "numbered", 3, (4, 7), 1, 2),
            (2, None, 1, "2", "2. Results", "numbered", 8, (9, 10, 11, 12, 13), 2, 3),
            (3, 2, 2, "2.1", "2.1 Details", "numbered", 16, (17,), 3, 3),
        ],
        [
            (
                0,
                0,
                "prose",
                "This report covers the year. Costs rose by 3.5 percent in total.",
                0,
                16,
                1,
                1,
                ((2, 0, 64),),
            ),
            (
                1,
                1,
                "prose",
                "The committee met twice. Its findings continue on the next page without a break. A new sentence ends it.",
                0,
                22,
                1,
                2,
                ((4, 0, 53), (7, 0, 50)),
            ),
            (
                2,
                2,
                "prose",
                "Left column first. Left column second.\n\nLeft column third.",
                0,
                12,
                2,
                2,
                ((9, 0, 38), (10, 0, 18)),
            ),
            # The two short paragraphs just above the table are its caption, and lead it.
            (
                3,
                2,
                "table",
                "Right column first. Right column second.\n\nRight column third.\n\n"
                "| Region | Sales |\n| North | 120 |\n| South | 95 |",
                0,
                27,
                2,
                2,
                ((11, 0, 40), (12, 0, 19), (13, 0, 49)),
            ),
            (4, 3, "prose", "Details close the report.", 0, 5, 3, 3, ((17, 0, 25),)),
        ],
        {
            "pages_without_text": [],
            "pages_probably_scanned": [],
            "blocks": 19,
            "tables": 1,
            "headers_and_footers": 6,
            "headings": 4,
            "bookmarks": 0,
            "section_source": "numbered",
        },
    )
    assert result == expected
    # The running header and footer never reach a chunk.
    exp_furniture = ("ACME Annual Report", "Page ")
    assert [
        any(part in chunk.text for part in exp_furniture) for chunk in tested.result(helper_report_pdf()).chunks
    ] == [False] * 5


def test_result__small_chunks_overlap_and_cross_pages() -> None:
    tested = helper_instance(helper_chunking(8, 14, 3, 1))
    result = helper_shapes(tested.result(helper_report_pdf()))[3]
    expected = [
        (0, 0, "prose", "This report covers the year.", 0, 6, 1, 1, ((2, 0, 28),)),
        # The previous sentence and this one would pass the maximum: no overlap.
        (1, 0, "prose", "Costs rose by 3.5 percent in total.", 0, 10, 1, 1, ((2, 29, 64),)),
        (2, 1, "prose", "The committee met twice.", 0, 5, 1, 1, ((4, 0, 24),)),
        (
            3,
            1,
            "prose",
            "Its findings continue on the next page without a break.",
            0,
            11,
            1,
            2,
            ((4, 25, 53), (7, 0, 26)),
        ),
        (4, 1, "prose", "A new sentence ends it.", 0, 6, 2, 2, ((7, 27, 50),)),
        (5, 2, "prose", "Left column first. Left column second.", 0, 8, 2, 2, ((9, 0, 38),)),
        (6, 2, "prose", "Left column second.\n\nLeft column third.", 21, 8, 2, 2, ((10, 0, 18),)),
        # The caption and header lead every chunk of the table; a chunk always gets its first row.
        (
            7,
            2,
            "table",
            "Right column first. Right column second.\n\nRight column third.\n\n| Region | Sales |\n| North | 120 |",
            0,
            22,
            2,
            2,
            ((11, 0, 40), (12, 0, 19), (13, 0, 34)),
        ),
        (
            8,
            2,
            "table",
            "Right column first. Right column second.\n\nRight column third.\n\n| Region | Sales |\n| South | 95 |",
            82,
            22,
            2,
            2,
            ((13, 35, 49),),
        ),
        (9, 3, "prose", "Details close the report.", 0, 5, 3, 3, ((17, 0, 25),)),
    ]
    assert result == expected


def test_result__no_heading_gives_one_untitled_section() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    data = (
        PdfFactory()
        .page()
        .paragraph("No heading here. Just text.", 10, False)
        .paragraph("A second paragraph follows.", 10, False)
        .page()
        .paragraph("Page two text.", 10, False)
        .built()
    )
    result = helper_shapes(tested.result(data))
    expected = (
        2,
        [
            (0, 1, "paragraph", "No heading here. Just text."),
            (1, 1, "paragraph", "A second paragraph follows."),
            (2, 2, "paragraph", "Page two text."),
        ],
        [(0, None, 1, "", "", "fallback", None, (0, 1, 2), 1, 2)],
        [
            (
                0,
                0,
                "prose",
                "No heading here. Just text.\n\nA second paragraph follows.\n\nPage two text.",
                0,
                16,
                1,
                2,
                ((0, 0, 27), (1, 0, 27), (2, 0, 14)),
            )
        ],
        {
            "pages_without_text": [],
            "pages_probably_scanned": [],
            "blocks": 3,
            "tables": 0,
            "headers_and_footers": 0,
            "headings": 0,
            "bookmarks": 0,
            "section_source": "fallback",
        },
    )
    assert result == expected


def test_result__bookmarks() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    data = (
        PdfFactory()
        .page()
        .paragraph("Overview", 10, False)
        .paragraph("The overview text.", 10, False)
        .page()
        .paragraph("Methods", 10, False)
        .paragraph("The methods text.", 10, False)
        .bookmarks([(1, "Overview", 1), (1, "Methods", 2)])
        .built()
    )
    result = helper_shapes(tested.result(data))
    expected = (
        2,
        [
            (0, 1, "heading", "Overview"),
            (1, 1, "paragraph", "The overview text."),
            (2, 2, "heading", "Methods"),
            (3, 2, "paragraph", "The methods text."),
        ],
        [
            (0, None, 1, "", "Overview", "bookmark", 0, (1,), 1, 1),
            (1, None, 1, "", "Methods", "bookmark", 2, (3,), 2, 2),
        ],
        [
            (0, 0, "prose", "The overview text.", 0, 4, 1, 1, ((1, 0, 18),)),
            (1, 1, "prose", "The methods text.", 0, 4, 2, 2, ((3, 0, 17),)),
        ],
        {
            "pages_without_text": [],
            "pages_probably_scanned": [],
            "blocks": 4,
            "tables": 0,
            "headers_and_footers": 0,
            "headings": 2,
            "bookmarks": 2,
            "section_source": "bookmark",
        },
    )
    assert result == expected


@pytest.mark.parametrize(
    "data, expected",
    [
        pytest.param(
            PdfFactory().page().image().page().image().built(),
            (
                2,
                [],
                [],
                [],
                {
                    "pages_without_text": [1, 2],
                    "pages_probably_scanned": [1, 2],
                    "blocks": 0,
                    "tables": 0,
                    "headers_and_footers": 0,
                    "headings": 0,
                    "bookmarks": 0,
                    "section_source": "",
                },
            ),
            id="scanned",
        ),
        pytest.param(
            PdfFactory().page().built(),
            (
                1,
                [],
                [],
                [],
                {
                    "pages_without_text": [1],
                    "pages_probably_scanned": [],
                    "blocks": 0,
                    "tables": 0,
                    "headers_and_footers": 0,
                    "headings": 0,
                    "bookmarks": 0,
                    "section_source": "",
                },
            ),
            id="blank",
        ),
    ],
)
def test_result__no_text(data: bytes, expected: tuple[Any, ...]) -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    processed = tested.result(data)
    result = helper_shapes(processed)
    assert result == expected
    assert processed.has_text() is False


def test_result__malformed() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    with pytest.raises(AppException) as exc_info:
        tested.result(b"%PDF-1.4 not really")
    expected = (422, "This PDF could not be read: the file is damaged or is not a PDF.")
    assert (exc_info.value.status_code, exc_info.value.message) == expected


def test_report() -> None:
    tested = ProcessingCommand
    block = RawBlock(page=1, x0=0, y0=0, x1=10, y1=10, lines=("Hi",), font_size=10.0, bold_share=0.0)
    document = PdfDocument(
        pages=(
            PdfPage(number=1, width=100, height=100, blocks=(block,), tables=(), image_count=2),
            PdfPage(number=2, width=100, height=100, blocks=(), tables=(), image_count=1),
            PdfPage(number=3, width=100, height=100, blocks=(), tables=(), image_count=0),
        ),
        outline=(OutlineEntry(level=1, title="Hi", page=1),),
    )
    kinds = [BlockKind.HEADER, BlockKind.FOOTER, BlockKind.HEADING, BlockKind.TABLE, BlockKind.PARAGRAPH]
    blocks = [
        TextBlock(
            block_id=index,
            page=1,
            kind=kind,
            text="x",
            font_size=10.0,
            bold_share=0.0,
            line_count=1,
            top=0.5,
            bottom=0.5,
        )
        for index, kind in enumerate(kinds)
    ]
    result = tested.report(document, blocks, "layout")
    expected = {
        "pages_without_text": [2, 3],
        "pages_probably_scanned": [2],
        "blocks": 5,
        "tables": 1,
        "headers_and_footers": 2,
        "headings": 1,
        "bookmarks": 1,
        "section_source": "layout",
    }
    assert result == expected


def test_reprocessed() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    with (
        patch.object(ProcessingStore, "requested", side_effect=[1]) as mock_requested,
        patch.object(ProcessingStore, "processing_of", side_effect=[]) as mock_processing_of,
        patch.object(ProcessingCommand, "queued", side_effect=[None]) as mock_queued,
    ):
        result = tested.reprocessed(helper_user(), 5)
    assert result is None
    exp_calls = [call(7, 5)]
    assert mock_requested.mock_calls == exp_calls
    exp_calls = []
    assert mock_processing_of.mock_calls == exp_calls
    exp_calls = [call(5)]
    assert mock_queued.mock_calls == exp_calls


@pytest.mark.parametrize(
    "found, expected",
    [
        pytest.param(None, (404, "This document does not exist."), id="not-the-owners"),
        pytest.param(
            helper_processing(MediaType.PNG),
            (400, "Only PDF files are processed for search for now."),
            id="not-a-pdf",
        ),
    ],
)
def test_reprocessed__refused(found: DocumentProcessing | None, expected: tuple[int, str]) -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    with (
        patch.object(ProcessingStore, "requested", side_effect=[0]) as mock_requested,
        patch.object(ProcessingStore, "processing_of", side_effect=[found]) as mock_processing_of,
        patch.object(ProcessingCommand, "queued", side_effect=[]) as mock_queued,
        pytest.raises(AppException) as exc_info,
    ):
        tested.reprocessed(helper_user(), 5)
    assert (exc_info.value.status_code, exc_info.value.message) == expected
    exp_calls = [call(7, 5)]
    assert mock_requested.mock_calls == exp_calls
    assert mock_processing_of.mock_calls == exp_calls
    exp_calls = []
    assert mock_queued.mock_calls == exp_calls


def test_processing() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    section = StoredSection(
        section_id=11,
        parent_id=None,
        position=0,
        level=1,
        number="1",
        title="1. Introduction",
        source=SectionSource.NUMBERED,
        start_page=1,
        end_page=1,
        heading_block_id=1,
        chunk_count=1,
    )
    chunk = StoredChunk(
        chunk_id=21,
        section_id=11,
        position=0,
        kind=ChunkKind.PROSE,
        text="Hello.",
        prefix_chars=0,
        token_count=2,
        start_page=1,
        end_page=1,
        spans=(SourceSpan(block_id=2, start=0, end=6),),
        strategy="sentences-v2 target=200 max=350 min=75 overlap=1",
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    with (
        patch.object(ProcessingStore, "processing_of", side_effect=[helper_processing(MediaType.PDF)]) as mock_of,
        patch.object(ProcessingStore, "sections_of", side_effect=[[section]]) as mock_sections_of,
        patch.object(ProcessingStore, "chunks_of", side_effect=[[chunk]]) as mock_chunks_of,
    ):
        result = tested.processing(helper_user(), 5)
    expected = (helper_processing(MediaType.PDF), [section], [chunk])
    assert result == expected
    exp_calls = [call(7, 5)]
    assert mock_of.mock_calls == exp_calls
    exp_calls = [call(5)]
    assert mock_sections_of.mock_calls == exp_calls
    assert mock_chunks_of.mock_calls == exp_calls


def test_processing__not_the_owners() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    with (
        patch.object(ProcessingStore, "processing_of", side_effect=[None]) as mock_of,
        patch.object(ProcessingStore, "sections_of", side_effect=[]) as mock_sections_of,
        patch.object(ProcessingStore, "chunks_of", side_effect=[]) as mock_chunks_of,
        pytest.raises(AppException) as exc_info,
    ):
        tested.processing(helper_user(), 5)
    expected = (404, "This document does not exist.")
    assert (exc_info.value.status_code, exc_info.value.message) == expected
    exp_calls = [call(7, 5)]
    assert mock_of.mock_calls == exp_calls
    exp_calls = []
    assert mock_sections_of.mock_calls == exp_calls
    assert mock_chunks_of.mock_calls == exp_calls


def test_extraction() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    block = TextBlock(
        block_id=0,
        page=1,
        kind=BlockKind.PARAGRAPH,
        text="Hello.",
        font_size=10.0,
        bold_share=0.0,
        line_count=1,
        top=0.1,
        bottom=0.2,
    )
    with (
        patch.object(ProcessingStore, "processing_of", side_effect=[helper_processing(MediaType.PDF)]) as mock_of,
        patch.object(ProcessingStore, "extraction_of", side_effect=[[block]]) as mock_extraction_of,
    ):
        result = tested.extraction(helper_user(), 5)
    expected = [block]
    assert result == expected
    exp_calls = [call(7, 5)]
    assert mock_of.mock_calls == exp_calls
    assert mock_extraction_of.mock_calls == exp_calls


def test_extraction__not_the_owners() -> None:
    tested = helper_instance(helper_chunking(200, 350, 75, 1))
    with (
        patch.object(ProcessingStore, "processing_of", side_effect=[None]) as mock_of,
        patch.object(ProcessingStore, "extraction_of", side_effect=[]) as mock_extraction_of,
        pytest.raises(AppException) as exc_info,
    ):
        tested.extraction(helper_user(), 5)
    expected = (404, "This document does not exist.")
    assert (exc_info.value.status_code, exc_info.value.message) == expected
    exp_calls = [call(7, 5)]
    assert mock_of.mock_calls == exp_calls
    exp_calls = []
    assert mock_extraction_of.mock_calls == exp_calls


def test_strategy() -> None:
    tested = helper_instance(helper_chunking(150, 300, 50, 2))
    result = tested.strategy()
    expected = "sentences-v2 target=150 max=300 min=50 overlap=2"
    assert result == expected
