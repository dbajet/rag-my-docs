from __future__ import annotations

import logging
import time
from typing import Any

from rag_my_docs.constants.constants import Constants
from rag_my_docs.constants.document_constants import DocumentConstants
from rag_my_docs.constants.embedding_constants import EmbeddingConstants
from rag_my_docs.libraries.embedding_box import EmbeddingBox
from rag_my_docs.libraries.index_store import IndexStore
from rag_my_docs.libraries.job_queue import JobQueue
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.chunk_to_embed import ChunkToEmbed
from rag_my_docs.structures.claimed_indexing import ClaimedIndexing
from rag_my_docs.structures.user_account import UserAccount


class IndexCommand:
    """The indexing stage, after extraction and chunking: every chunk of a
    processed document gets an embedding from the local model.

    A run takes the document, embeds only the chunks that have no embedding
    from the current model (an unchanged chunk keeps the one it had), stores
    them batch by batch, and marks the document indexed only once every chunk
    has one. A failure leaves the stored batches — they are valid — and marks
    the document failed, never indexed. It runs on a job thread of its own; the
    document's index status is the queue that survives a restart."""

    def __init__(self, store: IndexStore, box: EmbeddingBox, jobs: JobQueue) -> None:
        self._store = store
        self._box = box
        self._jobs = jobs

    def queued(self, document_id: int) -> None:
        self._jobs.queued(self.indexed, (document_id,))

    def resumed(self) -> None:
        """At startup: load the model on the job thread before any work, send
        back to the queue the runs a stopped process left and the documents
        indexed by another model or passage format, and queue everything waiting."""
        self._jobs.queued(self._box.warmed, ())
        self._store.restarted(EmbeddingConstants.stale_run_minutes)
        self._store.outdated(self._box.space())
        for document_id in self._store.pending_ids():
            self.queued(document_id)

    def indexed(self, document_id: int) -> None:
        if (claimed := self._store.claimed(document_id, self._box.space())) is None:
            return
        started = time.perf_counter()
        try:
            chunks = self._store.missing(claimed)
            batches = self.embedded(claimed, chunks)
        except AppException as error:
            self._store.failed(claimed, error.message, {})
            return
        except Exception:  # a fallback: the document must not stay "indexing" forever
            logging.getLogger(Constants.logger_name).exception("[INDEXING] document %s failed", document_id)
            self._store.failed(claimed, EmbeddingConstants.unexpected, {})
            raise
        self.closed(claimed, len(chunks), batches, started)

    def embedded(self, claimed: ClaimedIndexing, chunks: list[ChunkToEmbed]) -> int:
        """Embeds and stores the chunks batch by batch; the number of batches."""
        result = 0
        for batch in self.batches(chunks, self._box.batch_size()):
            vectors = self._box.passages([chunk.text for chunk in batch])
            self._store.embedded(claimed, batch, vectors)
            result += 1
        return result

    def closed(self, claimed: ClaimedIndexing, embedded: int, batches: int, started: float) -> None:
        total, done = self._store.counts(claimed)
        report: dict[str, Any] = {
            "chunks": total,
            "embedded": embedded,
            "reused": total - embedded,
            "batches": batches,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if total == 0:
            self._store.failed(claimed, EmbeddingConstants.not_indexable, report)
        elif done != total:
            self._store.failed(claimed, EmbeddingConstants.incomplete, report)
        elif self._store.finished(claimed, report):
            logging.getLogger(Constants.logger_name).info(
                "[INDEXING] document %s: %s chunks, %s embedded in %s ms",
                claimed.document_id,
                total,
                embedded,
                report["duration_ms"],
            )

    @classmethod
    def batches(cls, chunks: list[ChunkToEmbed], size: int) -> list[list[ChunkToEmbed]]:
        return [chunks[start : start + size] for start in range(0, len(chunks), size)]

    def reindexed(self, user: UserAccount, document_id: int, rebuild: bool) -> None:
        """Queues the owner's document; `rebuild` embeds every chunk again."""
        if self._store.requested(user.user_id, document_id, rebuild) == 0:
            if not self._store.exists(user.user_id, document_id):
                raise AppException(404, DocumentConstants.not_found)
            raise AppException(400, EmbeddingConstants.not_indexable)
        self.queued(document_id)

    def indexed_all(self, user: UserAccount) -> int:
        """Queues every processed document of the owner not yet indexed by the current model."""
        result = self._store.requested_all(user.user_id, self._box.space())
        for document_id in result:
            self.queued(document_id)
        return len(result)
