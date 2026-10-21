from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.passage_text import PassageText
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.libraries.vector_text import VectorText
from rag_my_docs.structures.chunk_to_embed import ChunkToEmbed
from rag_my_docs.structures.claimed_indexing import ClaimedIndexing


class IndexStore:
    """The indexing stage's state: which documents wait for embeddings, the
    embeddings themselves, and the run that holds a document. As with
    processing, a document is taken by one atomic UPDATE, and only the run
    that still holds it may mark it indexed — a reprocess or a reindex asked
    for meanwhile puts it back to pending, and wins."""

    def __init__(self, database: Database, sealer: SecretSealer, paths: SectionPaths) -> None:
        self._database = database
        self._sealer = sealer
        self._paths = paths

    def claimed(self, document_id: int, model: str) -> ClaimedIndexing | None:
        row = self._database.row(
            """UPDATE documents SET index_status = %s, index_run = index_run + 1, index_started_at = now(),
                   index_error = NULL, index_model = %s
               WHERE id = %s AND index_status = %s AND processing_status = %s
               RETURNING index_run""",
            (
                IndexStatus.INDEXING.value,
                model,
                document_id,
                IndexStatus.PENDING.value,
                ProcessingStatus.COMPLETED.value,
            ),
        )
        return None if row is None else ClaimedIndexing(document_id=document_id, run=int(row["index_run"]), model=model)

    def missing(self, claimed: ClaimedIndexing) -> list[ChunkToEmbed]:
        """The chunks without an embedding from this model, in document order,
        each written as the passage to embed."""
        rows = self._database.rows(
            """SELECT c.id, c.section_id, c.text_sealed, d.name_sealed
               FROM document_chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.document_id = %s AND (c.embedding IS NULL OR c.embedding_model IS DISTINCT FROM %s)
               ORDER BY c.position""",
            (claimed.document_id, claimed.model),
        )
        paths = self._paths.paths([claimed.document_id])
        return [
            ChunkToEmbed(
                chunk_id=int(row["id"]),
                text=PassageText.of(
                    self._sealer.opened(bytes(row["name_sealed"])),
                    paths[int(row["section_id"])],
                    self._sealer.opened(bytes(row["text_sealed"])),
                ),
            )
            for row in rows
        ]

    def embedded(self, claimed: ClaimedIndexing, chunks: list[ChunkToEmbed], vectors: list[list[float]]) -> None:
        """One batch, stored together. A chunk deleted meanwhile is simply not found."""
        with self._database.transaction():
            for chunk, vector in zip(chunks, vectors, strict=True):
                self._database.executed(
                    """UPDATE document_chunks SET embedding = %s::vector, embedding_model = %s
                       WHERE id = %s AND document_id = %s""",
                    (VectorText.of(vector), claimed.model, chunk.chunk_id, claimed.document_id),
                )

    def counts(self, claimed: ClaimedIndexing) -> tuple[int, int]:
        """The document's chunks, and those with an embedding from this model."""
        row = self._database.returned(
            """SELECT count(*) AS total,
                   count(*) FILTER (WHERE embedding IS NOT NULL AND embedding_model = %s) AS embedded
               FROM document_chunks WHERE document_id = %s""",
            (claimed.model, claimed.document_id),
        )
        return int(row["total"]), int(row["embedded"])

    def finished(self, claimed: ClaimedIndexing, report: dict[str, Any]) -> bool:
        return self._closed(claimed, IndexStatus.INDEXED, None, report)

    def failed(self, claimed: ClaimedIndexing, message: str, report: dict[str, Any]) -> bool:
        return self._closed(claimed, IndexStatus.FAILED, message, report)

    def _closed(self, claimed: ClaimedIndexing, status: IndexStatus, error: str | None, report: dict[str, Any]) -> bool:
        """Whether the run still held the document when it ended."""
        row = self._database.row(
            """UPDATE documents SET index_status = %s, index_error = %s, index_report = %s, indexed_at = now()
               WHERE id = %s AND index_run = %s AND index_status = %s
               RETURNING id""",
            (status.value, error, Jsonb(report), claimed.document_id, claimed.run, IndexStatus.INDEXING.value),
        )
        return row is not None

    def requested(self, user_id: int, document_id: int, rebuild: bool) -> int:
        """Puts the owner's processed document back in the queue; with `rebuild`,
        its embeddings are dropped first, so every chunk is embedded again."""
        with self._database.transaction():
            count = self._database.executed(
                """UPDATE documents SET index_status = %s
                   WHERE id = %s AND user_id = %s AND processing_status = %s""",
                (IndexStatus.PENDING.value, document_id, user_id, ProcessingStatus.COMPLETED.value),
            )
            if count and rebuild:
                self._database.executed(
                    "UPDATE document_chunks SET embedding = NULL, embedding_model = NULL WHERE document_id = %s",
                    (document_id,),
                )
        return count

    def requested_all(self, user_id: int, model: str) -> list[int]:
        """The owner's processed documents not indexed with this model, put back in the queue."""
        rows = self._database.rows(
            """UPDATE documents SET index_status = %s
               WHERE user_id = %s AND processing_status = %s
                 AND (index_status IN (%s, %s) OR (index_status = %s AND index_model IS DISTINCT FROM %s))
               RETURNING id""",
            (
                IndexStatus.PENDING.value,
                user_id,
                ProcessingStatus.COMPLETED.value,
                IndexStatus.NOT_INDEXED.value,
                IndexStatus.FAILED.value,
                IndexStatus.INDEXED.value,
                model,
            ),
        )
        return sorted(int(row["id"]) for row in rows)

    def exists(self, user_id: int, document_id: int) -> bool:
        return (
            self._database.row(
                "SELECT 1 AS found FROM documents WHERE id = %s AND user_id = %s", (document_id, user_id)
            )
            is not None
        )

    def outdated(self, model: str) -> int:
        """Documents indexed with another model go back to the queue: a vector
        is only comparable with vectors from the same model."""
        return self._database.executed(
            "UPDATE documents SET index_status = %s WHERE index_status = %s AND index_model IS DISTINCT FROM %s",
            (IndexStatus.PENDING.value, IndexStatus.INDEXED.value, model),
        )

    def pending_ids(self) -> list[int]:
        rows = self._database.rows(
            "SELECT id FROM documents WHERE index_status = %s ORDER BY id", (IndexStatus.PENDING.value,)
        )
        return [int(row["id"]) for row in rows]

    def restarted(self, minutes: int) -> int:
        """Runs started long ago by a process that has stopped go back to the queue."""
        return self._database.executed(
            """UPDATE documents SET index_status = %s
               WHERE index_status = %s AND index_started_at < now() - make_interval(mins => %s)""",
            (IndexStatus.PENDING.value, IndexStatus.INDEXING.value, minutes),
        )
