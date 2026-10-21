from __future__ import annotations

import json
from typing import Any

from psycopg.types.json import Jsonb

from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.passage_text import PassageText
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.libraries.text_digest import TextDigest
from rag_my_docs.structures.claimed_document import ClaimedDocument
from rag_my_docs.structures.document_processing import DocumentProcessing
from rag_my_docs.structures.embedding_carry import EmbeddingCarry
from rag_my_docs.structures.processing_result import ProcessingResult
from rag_my_docs.structures.source_span import SourceSpan
from rag_my_docs.structures.stored_chunk import StoredChunk
from rag_my_docs.structures.stored_section import StoredSection
from rag_my_docs.structures.text_block import TextBlock


class ProcessingStore:
    """A document's place in the pipeline, and what the pipeline made of it.

    Results replace the previous ones in one transaction — the old sections and
    chunks are deleted and the new ones written, or nothing changes — and only
    the run that still holds the document may write them: a document deleted,
    or sent back to the queue, while it was being processed keeps its new state.
    Section titles, chunk texts and the extracted blocks are sealed like the
    document itself; the spans, pages and counts are not text and stay clear.

    A chunk whose passage — its text and the section path above it — is
    unchanged by the new run keeps the embedding it had, matched by its keyed
    digest, so reprocessing only sends the changed chunks to the indexing
    stage; the document goes back to that stage's queue."""

    def __init__(self, database: Database, sealer: SecretSealer, digest: TextDigest) -> None:
        self._database = database
        self._sealer = sealer
        self._digest = digest

    def claimed(self, document_id: int) -> ClaimedDocument | None:
        """The document, taken for one run, if it was waiting; one atomic UPDATE,
        so two workers can never both take it."""
        row = self._database.row(
            """UPDATE documents SET processing_status = %s, processing_run = processing_run + 1,
                   processing_started_at = now(), processing_error = NULL
               WHERE id = %s AND processing_status = %s
               RETURNING id, processing_run, media_type, content_sealed""",
            (ProcessingStatus.PROCESSING.value, document_id, ProcessingStatus.PENDING.value),
        )
        if row is None:
            return None
        return ClaimedDocument(
            document_id=int(row["id"]),
            run=int(row["processing_run"]),
            media_type=MediaType(row["media_type"]),
            content=self._sealer.opened_bytes(bytes(row["content_sealed"])),
        )

    def finished(self, claimed: ClaimedDocument, result: ProcessingResult, strategy: str) -> bool:
        """Whether the results were stored: False when the run no longer holds the document."""
        extraction = json.dumps([block.to_dict() for block in result.blocks])
        with self._database.transaction():
            held = self._database.row(
                """UPDATE documents SET processing_status = %s, processing_error = NULL, page_count = %s,
                       chunking_version = %s, processed_at = now(), processing_report = %s, extraction_sealed = %s,
                       index_status = %s, index_error = NULL
                   WHERE id = %s AND processing_run = %s AND processing_status = %s
                   RETURNING name_sealed""",
                (
                    ProcessingStatus.COMPLETED.value,
                    result.page_count,
                    strategy,
                    Jsonb(result.report),
                    self._sealer.sealed(extraction),
                    IndexStatus.PENDING.value,
                    claimed.document_id,
                    claimed.run,
                    ProcessingStatus.PROCESSING.value,
                ),
            )
            if held is None:
                return False
            carried = self.embeddings_of(claimed.document_id)
            self._database.executed("DELETE FROM document_sections WHERE document_id = %s", (claimed.document_id,))
            section_ids = self.sections_written(claimed.document_id, result)
            name = self._sealer.opened(bytes(held["name_sealed"]))
            self.chunks_written(claimed.document_id, name, result, section_ids, strategy, carried)
        return True

    def embeddings_of(self, document_id: int) -> dict[str, EmbeddingCarry]:
        """The document's current embeddings, by the digest of their chunk's passage."""
        rows = self._database.rows(
            """SELECT text_digest, embedding::text AS embedding, embedding_model FROM document_chunks
               WHERE document_id = %s AND embedding IS NOT NULL AND text_digest IS NOT NULL""",
            (document_id,),
        )
        return {
            str(row["text_digest"]): EmbeddingCarry(vector=str(row["embedding"]), model=str(row["embedding_model"]))
            for row in rows
        }

    def sections_written(self, document_id: int, result: ProcessingResult) -> dict[int, int]:
        """The database id of each section, by its position."""
        section_ids: dict[int, int] = {}
        for section in result.sections:
            row = self._database.returned(
                """INSERT INTO document_sections (document_id, parent_id, position, level, number, title_sealed,
                       source, start_page, end_page, heading_block)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    document_id,
                    None if section.parent_position is None else section_ids[section.parent_position],
                    section.position,
                    section.level,
                    section.number,
                    self._sealer.sealed(section.title),
                    section.source.value,
                    section.start_page,
                    section.end_page,
                    section.heading_block_id,
                ),
            )
            section_ids[section.position] = int(row["id"])
        return section_ids

    def chunks_written(
        self,
        document_id: int,
        name: str,
        result: ProcessingResult,
        section_ids: dict[int, int],
        strategy: str,
        carried: dict[str, EmbeddingCarry],
    ) -> None:
        paths = SectionPaths.joined(
            {section.position: section.title for section in result.sections},
            {section.position: section.parent_position for section in result.sections},
        )
        for chunk in result.chunks:
            digest = self._digest.digest(PassageText.of(name, paths[chunk.section_position], chunk.text))
            kept = carried.get(digest)
            self._database.executed(
                """INSERT INTO document_chunks (document_id, section_id, position, kind, text_sealed, prefix_chars,
                       token_count, start_page, end_page, source_spans, strategy, text_digest, embedding, embedding_model)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)""",
                (
                    document_id,
                    section_ids[chunk.section_position],
                    chunk.position,
                    chunk.kind.value,
                    self._sealer.sealed(chunk.text),
                    chunk.prefix_chars,
                    chunk.token_count,
                    chunk.start_page,
                    chunk.end_page,
                    Jsonb([span.to_dict() for span in chunk.spans]),
                    strategy,
                    digest,
                    None if kept is None else kept.vector,
                    None if kept is None else kept.model,
                ),
            )

    def closed(
        self,
        claimed: ClaimedDocument,
        status: ProcessingStatus,
        error: str,
        report: dict[str, Any],
    ) -> bool:
        """Ends a run without results (failed, no text, unsupported): the
        previous results go too, so nothing stale is left to search."""
        with self._database.transaction():
            held = self._database.row(
                """UPDATE documents SET processing_status = %s, processing_error = %s, page_count = NULL,
                       chunking_version = NULL, processed_at = now(), processing_report = %s, extraction_sealed = NULL,
                       index_status = %s, index_error = NULL
                   WHERE id = %s AND processing_run = %s AND processing_status = %s
                   RETURNING id""",
                (
                    status.value,
                    error,
                    Jsonb(report),
                    IndexStatus.NOT_INDEXED.value,
                    claimed.document_id,
                    claimed.run,
                    ProcessingStatus.PROCESSING.value,
                ),
            )
            if held is None:
                return False
            self._database.executed("DELETE FROM document_sections WHERE document_id = %s", (claimed.document_id,))
        return True

    def requested(self, user_id: int, document_id: int) -> int:
        """Puts the owner's PDF back in the queue; rows touched (0: not theirs, or not a PDF)."""
        return self._database.executed(
            "UPDATE documents SET processing_status = %s WHERE id = %s AND user_id = %s AND media_type = %s",
            (ProcessingStatus.PENDING.value, document_id, user_id, MediaType.PDF.value),
        )

    def pending_ids(self) -> list[int]:
        rows = self._database.rows(
            "SELECT id FROM documents WHERE processing_status = %s ORDER BY id", (ProcessingStatus.PENDING.value,)
        )
        return [int(row["id"]) for row in rows]

    def restarted(self, minutes: int) -> int:
        """Puts back in the queue the runs started long ago by a process that
        has stopped; a run of the other live color is younger than that."""
        return self._database.executed(
            """UPDATE documents SET processing_status = %s
               WHERE processing_status = %s AND processing_started_at < now() - make_interval(mins => %s)""",
            (ProcessingStatus.PENDING.value, ProcessingStatus.PROCESSING.value, minutes),
        )

    def processing_of(self, user_id: int, document_id: int) -> DocumentProcessing | None:
        row = self._database.row(
            """SELECT id, name_sealed, media_type, processing_status, processing_error, page_count, chunking_version,
                   to_char(processed_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS processed_at,
                   processing_report, index_status, index_model, index_error, index_report,
                   to_char(indexed_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS indexed_at,
                   (SELECT count(*) FROM document_chunks c WHERE c.document_id = documents.id) AS chunk_count,
                   (SELECT count(*) FROM document_chunks c
                     WHERE c.document_id = documents.id AND c.embedding IS NOT NULL) AS indexed_count
               FROM documents WHERE id = %s AND user_id = %s""",
            (document_id, user_id),
        )
        if row is None:
            return None
        return DocumentProcessing(
            document_id=int(row["id"]),
            name=self._sealer.opened(bytes(row["name_sealed"])),
            media_type=MediaType(row["media_type"]),
            status=ProcessingStatus(row["processing_status"]),
            error=row["processing_error"] or "",
            page_count=row["page_count"],
            chunking_version=row["chunking_version"] or "",
            processed_at=row["processed_at"] or "",
            report=row["processing_report"] or {},
            index_status=IndexStatus(row["index_status"]),
            index_model=row["index_model"] or "",
            index_error=row["index_error"] or "",
            indexed_at=row["indexed_at"] or "",
            index_report=row["index_report"] or {},
            chunk_count=int(row["chunk_count"]),
            indexed_count=int(row["indexed_count"]),
        )

    def sections_of(self, document_id: int) -> list[StoredSection]:
        rows = self._database.rows(
            """SELECT s.id, s.parent_id, s.position, s.level, s.number, s.title_sealed, s.source, s.start_page,
                   s.end_page, s.heading_block, (SELECT count(*) FROM document_chunks c WHERE c.section_id = s.id) AS chunks
               FROM document_sections s WHERE s.document_id = %s ORDER BY s.position""",
            (document_id,),
        )
        return [
            StoredSection(
                section_id=int(row["id"]),
                parent_id=None if row["parent_id"] is None else int(row["parent_id"]),
                position=int(row["position"]),
                level=int(row["level"]),
                number=str(row["number"]),
                title=self._sealer.opened(bytes(row["title_sealed"])),
                source=SectionSource(row["source"]),
                start_page=int(row["start_page"]),
                end_page=int(row["end_page"]),
                heading_block_id=None if row["heading_block"] is None else int(row["heading_block"]),
                chunk_count=int(row["chunks"]),
            )
            for row in rows
        ]

    def chunks_of(self, document_id: int) -> list[StoredChunk]:
        rows = self._database.rows(
            """SELECT id, section_id, position, kind, text_sealed, prefix_chars, token_count, start_page, end_page,
                   source_spans, strategy, embedding_model
               FROM document_chunks WHERE document_id = %s ORDER BY position""",
            (document_id,),
        )
        return [
            StoredChunk(
                chunk_id=int(row["id"]),
                section_id=int(row["section_id"]),
                position=int(row["position"]),
                kind=ChunkKind(row["kind"]),
                text=self._sealer.opened(bytes(row["text_sealed"])),
                prefix_chars=int(row["prefix_chars"]),
                token_count=int(row["token_count"]),
                start_page=int(row["start_page"]),
                end_page=int(row["end_page"]),
                spans=tuple(SourceSpan.from_dict(span) for span in row["source_spans"]),
                strategy=str(row["strategy"]),
                embedding_model=row["embedding_model"] or "",
            )
            for row in rows
        ]

    def extraction_of(self, user_id: int, document_id: int) -> list[TextBlock]:
        """The extracted blocks of the owner's document; none when it has no results."""
        row = self._database.row(
            "SELECT extraction_sealed FROM documents WHERE id = %s AND user_id = %s", (document_id, user_id)
        )
        if row is None or row["extraction_sealed"] is None:
            return []
        blocks: list[dict[str, Any]] = json.loads(self._sealer.opened(bytes(row["extraction_sealed"])))
        return [TextBlock.from_dict(block) for block in blocks]
