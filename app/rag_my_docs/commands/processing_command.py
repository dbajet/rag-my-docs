from __future__ import annotations

import logging
from typing import Any

from rag_my_docs.commands.index_command import IndexCommand
from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.constants.constants import Constants
from rag_my_docs.constants.document_constants import DocumentConstants
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_constants import ProcessingConstants
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.libraries.chunker import Chunker
from rag_my_docs.libraries.job_queue import JobQueue
from rag_my_docs.libraries.page_furniture import PageFurniture
from rag_my_docs.libraries.page_layout import PageLayout
from rag_my_docs.libraries.pdf_box import PdfBox
from rag_my_docs.libraries.processing_store import ProcessingStore
from rag_my_docs.libraries.section_detector import SectionDetector
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.claimed_document import ClaimedDocument
from rag_my_docs.structures.document_processing import DocumentProcessing
from rag_my_docs.structures.pdf_document import PdfDocument
from rag_my_docs.structures.processing_result import ProcessingResult
from rag_my_docs.structures.stored_chunk import StoredChunk
from rag_my_docs.structures.stored_section import StoredSection
from rag_my_docs.structures.text_block import TextBlock
from rag_my_docs.structures.user_account import UserAccount


class ProcessingCommand:
    """The pipeline, stage by stage: read the PDF, order its blocks, mark page
    headers and footers, detect sections, cut chunks — then store all of it at
    once. Each stage is its own class, so each can be tested and explained alone.

    It runs on a job thread of its own, so the app serves while a
    publication is read; the document's status is the queue that survives a restart: a pending document
    is picked up again when the application starts. The logs name documents by
    id and count things; they never quote a document."""

    def __init__(
        self,
        store: ProcessingStore,
        pdf_box: PdfBox,
        layout: PageLayout,
        furniture: PageFurniture,
        detector: SectionDetector,
        chunker: Chunker,
        strategy: str,
        jobs: JobQueue,
        indexing: IndexCommand,
    ) -> None:
        self._store = store
        self._pdf_box = pdf_box
        self._layout = layout
        self._furniture = furniture
        self._detector = detector
        self._chunker = chunker
        self._strategy = strategy
        self._jobs = jobs
        self._indexing = indexing

    def queued(self, document_id: int) -> None:
        self._jobs.queued(self.processed, (document_id,))

    def resumed(self) -> None:
        """At startup: runs a stopped process left behind go back to the queue,
        and everything waiting is queued."""
        self._store.restarted(ProcessingConstants.stale_run_minutes)
        for document_id in self._store.pending_ids():
            self.queued(document_id)

    def processed(self, document_id: int) -> None:
        if (claimed := self._store.claimed(document_id)) is None:
            return
        if claimed.media_type is not MediaType.PDF:
            self._store.closed(claimed, ProcessingStatus.UNSUPPORTED, ProcessingConstants.not_pdf, {})
            return
        try:
            result = self.result(claimed.content)
        except AppException as error:
            self._store.closed(claimed, ProcessingStatus.FAILED, error.message, {})
            return
        except Exception:  # a fallback: the document must not stay "processing" forever
            logging.getLogger(Constants.logger_name).exception("[PROCESSING] document %s failed", document_id)
            self._store.closed(claimed, ProcessingStatus.FAILED, ProcessingConstants.unexpected, {})
            raise
        self.stored(claimed, result)

    def stored(self, claimed: ClaimedDocument, result: ProcessingResult) -> None:
        if not result.has_text():
            self._store.closed(claimed, ProcessingStatus.NO_TEXT, ProcessingConstants.no_text, result.report)
            return
        if self._store.finished(claimed, result, self._strategy):
            # The next stage, on its own thread: embedding the new chunks.
            self._indexing.queued(claimed.document_id)
            logging.getLogger(Constants.logger_name).info(
                "[PROCESSING] document %s: %s pages, %s sections, %s chunks",
                claimed.document_id,
                result.page_count,
                len(result.sections),
                len(result.chunks),
            )

    def result(self, content: bytes) -> ProcessingResult:
        document = self._pdf_box.read(content)
        blocks = self._furniture.marked(self._layout.blocks(document))
        blocks, sections = self._detector.detected(blocks, document.outline)
        chunks = self._chunker.chunks(blocks, sections)
        return ProcessingResult(
            page_count=len(document.pages),
            blocks=tuple(blocks),
            sections=tuple(sections),
            chunks=tuple(chunks),
            report=self.report(document, blocks, sections[0].source.value if sections else ""),
        )

    @classmethod
    def report(cls, document: PdfDocument, blocks: list[TextBlock], section_source: str) -> dict[str, Any]:
        """What the inspection view shows about the extraction: numbers only."""
        empty = [page.number for page in document.pages if page.char_count() == 0]
        kinds = [block.kind for block in blocks]
        return {
            "pages_without_text": empty,
            "pages_probably_scanned": [
                page.number for page in document.pages if page.char_count() == 0 and page.image_count > 0
            ],
            "blocks": len(blocks),
            "tables": kinds.count(BlockKind.TABLE),
            "headers_and_footers": kinds.count(BlockKind.HEADER) + kinds.count(BlockKind.FOOTER),
            "headings": kinds.count(BlockKind.HEADING),
            "bookmarks": len(document.outline),
            "section_source": section_source,
        }

    def reprocessed(self, user: UserAccount, document_id: int) -> None:
        if self._store.requested(user.user_id, document_id) == 0:
            if self._store.processing_of(user.user_id, document_id) is None:
                raise AppException(404, DocumentConstants.not_found)
            raise AppException(400, ProcessingConstants.not_pdf)
        self.queued(document_id)

    def processing(
        self, user: UserAccount, document_id: int
    ) -> tuple[DocumentProcessing, list[StoredSection], list[StoredChunk]]:
        found = self._store.processing_of(user.user_id, document_id)
        if found is None:
            raise AppException(404, DocumentConstants.not_found)
        return found, self._store.sections_of(document_id), self._store.chunks_of(document_id)

    def extraction(self, user: UserAccount, document_id: int) -> list[TextBlock]:
        if self._store.processing_of(user.user_id, document_id) is None:
            raise AppException(404, DocumentConstants.not_found)
        return self._store.extraction_of(user.user_id, document_id)

    def strategy(self) -> str:
        return self._strategy
