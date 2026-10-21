from __future__ import annotations

from typing import Any

import pytest

from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.document_store import DocumentStore
from rag_my_docs.libraries.index_store import IndexStore
from rag_my_docs.libraries.processing_store import ProcessingStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.libraries.text_digest import TextDigest
from rag_my_docs.structures.chunk import Chunk
from rag_my_docs.structures.chunk_to_embed import ChunkToEmbed
from rag_my_docs.structures.claimed_indexing import ClaimedIndexing
from rag_my_docs.structures.processing_result import ProcessingResult
from rag_my_docs.structures.section import Section
from rag_my_docs.structures.source_span import SourceSpan
from tests.conftest import helper_database

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="
STRATEGY = "sentences-v1 target=200 max=350 min=75 overlap=1"
MODEL = "BAAI/bge-small-en-v1.5"
OTHER_MODEL = "intfloat/e5-small-v2"


def helper_chunk(position: int, text: str) -> Chunk:
    return Chunk(
        position=position,
        section_position=0,
        kind=ChunkKind.PROSE,
        text=text,
        prefix_chars=0,
        token_count=3,
        start_page=1,
        end_page=1,
        spans=(SourceSpan(block_id=0, start=0, end=len(text)),),
    )


def helper_result(texts: tuple[str, ...]) -> ProcessingResult:
    return ProcessingResult(
        page_count=1,
        blocks=(),
        sections=(
            Section(
                position=0,
                parent_position=None,
                level=1,
                number="",
                title="",
                source=SectionSource.FALLBACK,
                heading_block_id=None,
                block_ids=(),
                start_page=1,
                end_page=1,
            ),
        ),
        chunks=tuple(helper_chunk(position, text) for position, text in enumerate(texts)),
        report={},
    )


def helper_documents() -> Database:
    """Ada (1) and Bob (2). Document 1, Ada's, processed into chunks 1 and 2;
    document 2, Ada's, waiting for extraction; document 3, Bob's, processed
    into chunk 3. Documents 1 and 3 wait for indexing."""
    result = helper_database()
    for name in ("ada", "bob"):
        result.executed(
            "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)",
            (b"sealed", f"hash-{name}", name),
        )
    sealer = SecretSealer(KEY)
    documents = DocumentStore(result, sealer)
    documents.created(1, "report.pdf", MediaType.PDF, b"%PDF-1.7 a", ProcessingStatus.PENDING)
    documents.created(1, "later.pdf", MediaType.PDF, b"%PDF-1.7 b", ProcessingStatus.PENDING)
    documents.created(2, "bob.pdf", MediaType.PDF, b"%PDF-1.7 c", ProcessingStatus.PENDING)
    processing = ProcessingStore(result, sealer, TextDigest(KEY))
    for document_id, texts in ((1, ("Sales grew.", "Costs rose.")), (3, ("Bob's notes.",))):
        claimed = processing.claimed(document_id)
        assert claimed is not None
        processing.finished(claimed, helper_result(texts), STRATEGY)
    return result


def helper_store(database: Database) -> IndexStore:
    return IndexStore(database, SecretSealer(KEY), SectionPaths(database, SecretSealer(KEY)))


def helper_vector(axis: int) -> list[float]:
    """A unit vector along one of the 384 axes."""
    return [1.0 if index == axis else 0.0 for index in range(384)]


def helper_printed(axis: int) -> str:
    """The same vector as pgvector prints it."""
    return "[" + ",".join("1" if index == axis else "0" for index in range(384)) + "]"


def helper_to_embed(chunk_id: int) -> ChunkToEmbed:
    return ChunkToEmbed(chunk_id=chunk_id, text="Text.")


def helper_states(database: Database) -> list[dict[str, Any]]:
    return database.rows(
        """SELECT id, index_status, index_run, index_model, index_error, index_report,
                  index_started_at IS NOT NULL AS started, indexed_at IS NOT NULL AS ended
           FROM documents ORDER BY id""",
        (),
    )


def helper_embeddings(database: Database) -> list[dict[str, Any]]:
    return database.rows(
        "SELECT id, embedding::text AS embedding, embedding_model FROM document_chunks ORDER BY id", ()
    )


def helper_state(
    document_id: int,
    status: str,
    run: int,
    model: str | None,
    error: str | None,
    report: dict[str, Any] | None,
    started: bool,
    ended: bool,
) -> dict[str, Any]:
    return {
        "id": document_id,
        "index_status": status,
        "index_run": run,
        "index_model": model,
        "index_error": error,
        "index_report": report,
        "started": started,
        "ended": ended,
    }


def helper_waiting(document_id: int) -> dict[str, Any]:
    return helper_state(document_id, "pending", 0, None, None, None, False, False)


def test___init__() -> None:
    database = Database("postgresql://u:p@h/d")
    sealer = SecretSealer(KEY)
    paths = SectionPaths(database, sealer)
    tested = IndexStore(database, sealer, paths)
    result = (tested._database is database, tested._sealer is sealer, tested._paths is paths)
    expected = (True, True, True)
    assert result == expected


@pytest.mark.database
def test_claimed() -> None:
    """Taken once by one atomic UPDATE; a document not yet extracted is not taken."""
    database = helper_documents()
    tested = helper_store(database)
    result = (tested.claimed(1, MODEL), tested.claimed(1, MODEL), tested.claimed(2, MODEL), tested.claimed(99, MODEL))
    expected = (ClaimedIndexing(document_id=1, run=1, model=MODEL), None, None, None)
    assert result == expected
    exp_states = [
        helper_state(1, "indexing", 1, MODEL, None, None, True, False),
        helper_state(2, "not_indexed", 0, None, None, None, False, False),
        helper_waiting(3),
    ]
    assert helper_states(database) == exp_states


@pytest.mark.database
def test_missing() -> None:
    """In document order, each written as its passage — the document's name and
    section path above the text; a chunk embedded by another model is missing too."""
    database = helper_documents()
    tested = helper_store(database)
    claimed = ClaimedIndexing(document_id=1, run=1, model=MODEL)
    before = tested.missing(claimed)
    database.executed(
        "UPDATE document_chunks SET embedding = %s::vector, embedding_model = %s WHERE id = %s",
        (helper_printed(0), MODEL, 1),
    )
    database.executed(
        "UPDATE document_chunks SET embedding = %s::vector, embedding_model = %s WHERE id = %s",
        (helper_printed(1), OTHER_MODEL, 2),
    )
    after = tested.missing(claimed)
    result = (before, after)
    expected = (
        [
            ChunkToEmbed(chunk_id=1, text="report › Untitled section\n\nSales grew."),
            ChunkToEmbed(chunk_id=2, text="report › Untitled section\n\nCosts rose."),
        ],
        [ChunkToEmbed(chunk_id=2, text="report › Untitled section\n\nCosts rose.")],
    )
    assert result == expected


@pytest.mark.database
def test_embedded() -> None:
    """Stored with the model's name; a chunk of another document is not touched."""
    database = helper_documents()
    tested = helper_store(database)
    claimed = ClaimedIndexing(document_id=1, run=1, model=MODEL)
    chunks = [ChunkToEmbed(chunk_id=2, text="Sales grew. Sales 2016"), ChunkToEmbed(chunk_id=3, text="Bob's.")]
    result = tested.embedded(claimed, chunks, [helper_vector(5), helper_vector(6)])
    assert result is None
    expected = [
        {"id": 1, "embedding": None, "embedding_model": None},
        {"id": 2, "embedding": helper_printed(5), "embedding_model": MODEL},
        {"id": 3, "embedding": None, "embedding_model": None},
    ]
    assert helper_embeddings(database) == expected


@pytest.mark.database
def test_embedded__a_mismatched_batch_is_refused() -> None:
    database = helper_documents()
    tested = helper_store(database)
    claimed = ClaimedIndexing(document_id=1, run=1, model=MODEL)
    with pytest.raises(ValueError) as exc_info:
        tested.embedded(claimed, [helper_to_embed(1), helper_to_embed(2)], [helper_vector(5)])
    result = str(exc_info.value)
    expected = "zip() argument 2 is shorter than argument 1"
    assert result == expected
    exp_embeddings = [
        {"id": 1, "embedding": None, "embedding_model": None},
        {"id": 2, "embedding": None, "embedding_model": None},
        {"id": 3, "embedding": None, "embedding_model": None},
    ]
    assert helper_embeddings(database) == exp_embeddings


@pytest.mark.database
def test_counts() -> None:
    """Only embeddings from the run's model count as done."""
    database = helper_documents()
    tested = helper_store(database)
    claimed = ClaimedIndexing(document_id=1, run=1, model=MODEL)
    before = tested.counts(claimed)
    tested.embedded(claimed, [helper_to_embed(1)], [helper_vector(0)])
    tested.embedded(claimed._replace(model=OTHER_MODEL), [helper_to_embed(2)], [helper_vector(1)])
    after = tested.counts(claimed)
    result = (before, after, tested.counts(claimed._replace(document_id=2)))
    expected = ((2, 0), (2, 1), (0, 0))
    assert result == expected


@pytest.mark.database
def test_finished() -> None:
    database = helper_documents()
    tested = helper_store(database)
    claimed = tested.claimed(1, MODEL)
    assert claimed is not None
    result = (tested.finished(claimed, {"chunks": 2}), tested.finished(claimed, {"chunks": 2}))
    expected = (True, False)
    assert result == expected
    exp_state = helper_state(1, "indexed", 1, MODEL, None, {"chunks": 2}, True, True)
    assert helper_states(database)[0] == exp_state


@pytest.mark.database
def test_failed() -> None:
    database = helper_documents()
    tested = helper_store(database)
    claimed = tested.claimed(1, MODEL)
    assert claimed is not None
    result = tested.failed(claimed, "Some chunks have no embedding after indexing.", {"chunks": 2})
    assert result is True
    exp_state = helper_state(
        1, "failed", 1, MODEL, "Some chunks have no embedding after indexing.", {"chunks": 2}, True, True
    )
    assert helper_states(database)[0] == exp_state


@pytest.mark.database
def test__closed() -> None:
    """Only the run that still holds the document may close it: a reindex
    asked for meanwhile, or a later run, wins."""
    database = helper_documents()
    tested = helper_store(database)
    claimed = tested.claimed(1, MODEL)
    assert claimed is not None
    stale = tested._closed(claimed._replace(run=0), IndexStatus.INDEXED, None, {})
    tested.requested(1, 1, False)
    sent_back = tested._closed(claimed, IndexStatus.INDEXED, None, {})
    result = (stale, sent_back)
    expected = (False, False)
    assert result == expected
    exp_state = helper_state(1, "pending", 1, MODEL, None, None, True, False)
    assert helper_states(database)[0] == exp_state


@pytest.mark.database
def test_requested() -> None:
    """The owner's processed document only; a rebuild drops its embeddings."""
    database = helper_documents()
    tested = helper_store(database)
    claimed = ClaimedIndexing(document_id=1, run=1, model=MODEL)
    tested.embedded(claimed, [helper_to_embed(1), helper_to_embed(2)], [helper_vector(0), helper_vector(1)])
    tested.embedded(claimed._replace(document_id=3), [helper_to_embed(3)], [helper_vector(2)])
    database.executed("UPDATE documents SET index_status = 'indexed'", ())
    kept = tested.requested(1, 1, False)
    embeddings_kept = helper_embeddings(database)
    result = (kept, tested.requested(1, 1, True), tested.requested(1, 3, True), tested.requested(1, 2, False))
    expected = (1, 1, 0, 0)
    assert result == expected
    exp_kept = [
        {"id": 1, "embedding": helper_printed(0), "embedding_model": MODEL},
        {"id": 2, "embedding": helper_printed(1), "embedding_model": MODEL},
        {"id": 3, "embedding": helper_printed(2), "embedding_model": MODEL},
    ]
    assert embeddings_kept == exp_kept
    exp_embeddings = [
        {"id": 1, "embedding": None, "embedding_model": None},
        {"id": 2, "embedding": None, "embedding_model": None},
        {"id": 3, "embedding": helper_printed(2), "embedding_model": MODEL},
    ]
    assert helper_embeddings(database) == exp_embeddings
    # Document 2 is not processed: it stays as it was.
    exp_statuses = ["pending", "indexed", "indexed"]
    assert [row["index_status"] for row in helper_states(database)] == exp_statuses


@pytest.mark.database
def test_requested_all() -> None:
    """Not indexed, failed, or indexed by another model: queued again. Indexed
    by this model, waiting, or someone else's: left alone."""
    database = helper_documents()
    tested = helper_store(database)
    for document_id in (4, 5, 6, 7):
        DocumentStore(database, SecretSealer(KEY)).created(
            1, f"{document_id}.pdf", MediaType.PDF, b"%PDF-1.7", ProcessingStatus.PENDING
        )
    database.executed("UPDATE documents SET processing_status = 'completed' WHERE id >= 4", ())
    for document_id, status, model in (
        (4, "not_indexed", None),
        (5, "failed", MODEL),
        (6, "indexed", OTHER_MODEL),
        (7, "indexed", MODEL),
    ):
        database.executed(
            "UPDATE documents SET index_status = %s, index_model = %s WHERE id = %s", (status, model, document_id)
        )
    result = tested.requested_all(1, MODEL)
    expected = [4, 5, 6]
    assert result == expected
    exp_statuses = ["pending", "not_indexed", "pending", "pending", "pending", "pending", "indexed"]
    assert [row["index_status"] for row in helper_states(database)] == exp_statuses


@pytest.mark.database
def test_exists() -> None:
    database = helper_documents()
    tested = helper_store(database)
    result = (tested.exists(1, 1), tested.exists(1, 3), tested.exists(2, 3), tested.exists(1, 99))
    expected = (True, False, True, False)
    assert result == expected


@pytest.mark.database
def test_outdated() -> None:
    database = helper_documents()
    tested = helper_store(database)
    database.executed("UPDATE documents SET index_status = 'indexed', index_model = %s WHERE id = 1", (OTHER_MODEL,))
    database.executed("UPDATE documents SET index_status = 'indexed', index_model = %s WHERE id = 3", (MODEL,))
    result = tested.outdated(MODEL)
    expected = 1
    assert result == expected
    exp_statuses = ["pending", "not_indexed", "indexed"]
    assert [row["index_status"] for row in helper_states(database)] == exp_statuses


@pytest.mark.database
def test_pending_ids() -> None:
    database = helper_documents()
    tested = helper_store(database)
    result = tested.pending_ids()
    expected = [1, 3]
    assert result == expected


@pytest.mark.database
def test_restarted() -> None:
    """A run older than the limit goes back to the queue; a recent one stays."""
    database = helper_documents()
    tested = helper_store(database)
    tested.claimed(1, MODEL)
    tested.claimed(3, MODEL)
    database.executed("UPDATE documents SET index_started_at = now() - interval '11 minutes' WHERE id = 1", ())
    result = tested.restarted(10)
    expected = 1
    assert result == expected
    exp_statuses = ["pending", "not_indexed", "indexing"]
    assert [row["index_status"] for row in helper_states(database)] == exp_statuses
