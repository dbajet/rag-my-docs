from __future__ import annotations

import json

import pytest

from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.document_store import DocumentStore
from rag_my_docs.libraries.processing_store import ProcessingStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.text_digest import TextDigest
from rag_my_docs.structures.chunk import Chunk
from rag_my_docs.structures.claimed_document import ClaimedDocument
from rag_my_docs.structures.document_processing import DocumentProcessing
from rag_my_docs.structures.embedding_carry import EmbeddingCarry
from rag_my_docs.structures.processing_result import ProcessingResult
from rag_my_docs.structures.section import Section
from rag_my_docs.structures.source_span import SourceSpan
from rag_my_docs.structures.stored_chunk import StoredChunk
from rag_my_docs.structures.stored_section import StoredSection
from rag_my_docs.structures.text_block import TextBlock
from tests.conftest import helper_database

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="
STRATEGY = "sentences-v1 target=200 max=350 min=75 overlap=1"
MODEL = "BAAI/bge-small-en-v1.5"
# The keyed digests of the two chunks' passages, computed with `openssl dgst -mac HMAC`:
# "report › 1. Intro\n\nFirst sentence. Second sentence." and
# "report › 1. Intro › Figures\n\n| A | B |\n| 1 | 2 |".
DIGEST_PROSE = "418beffc63630b571fcfd32e346ba177450dbd4682bdcac460bf08e3fb62aa14"
DIGEST_TABLE = "d27aff9acde8a1c56b5e99fb3008465e077fa5660d888f1838d52762b23a0855"


def helper_documents() -> Database:
    """Ada (1) and Bob (2); documents 1 (Ada's PDF, pending), 2 (Ada's image,
    unsupported) and 3 (Bob's PDF, pending)."""
    result = helper_database()
    for name in ("ada", "bob"):
        result.executed(
            "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)",
            (b"sealed", f"hash-{name}", name),
        )
    documents = DocumentStore(result, SecretSealer(KEY))
    documents.created(1, "report.pdf", MediaType.PDF, b"%PDF-1.7 report", ProcessingStatus.PENDING)
    documents.created(1, "photo.png", MediaType.PNG, b"\x89PNG", ProcessingStatus.UNSUPPORTED)
    documents.created(2, "bob.pdf", MediaType.PDF, b"%PDF-1.7 bob", ProcessingStatus.PENDING)
    return result


def helper_store(database: Database) -> ProcessingStore:
    return ProcessingStore(database, SecretSealer(KEY), TextDigest(KEY))


def helper_vector(axis: int) -> str:
    """A unit vector along one of the 384 axes, as pgvector prints it."""
    values = ["0"] * 384
    values[axis] = "1"
    return "[" + ",".join(values) + "]"


def helper_embed(database: Database, position: int, axis: int, model: str) -> None:
    database.executed(
        "UPDATE document_chunks SET embedding = %s::vector, embedding_model = %s WHERE position = %s",
        (helper_vector(axis), model, position),
    )


def helper_block(block_id: int, kind: BlockKind, text: str) -> TextBlock:
    return TextBlock(
        block_id=block_id,
        page=1,
        kind=kind,
        text=text,
        font_size=11.0,
        bold_share=0.0,
        line_count=1,
        top=0.2,
        bottom=0.3,
    )


def helper_result() -> ProcessingResult:
    """A heading, a paragraph and a table: two nested sections, two chunks."""
    return ProcessingResult(
        page_count=2,
        blocks=(
            helper_block(0, BlockKind.HEADING, "1. Intro"),
            helper_block(1, BlockKind.PARAGRAPH, "First sentence. Second sentence."),
            helper_block(2, BlockKind.TABLE, "| A | B |\n| 1 | 2 |"),
        ),
        sections=(
            Section(
                position=0,
                parent_position=None,
                level=1,
                number="1",
                title="1. Intro",
                source=SectionSource.NUMBERED,
                heading_block_id=0,
                block_ids=(1,),
                start_page=1,
                end_page=2,
            ),
            Section(
                position=1,
                parent_position=0,
                level=2,
                number="",
                title="Figures",
                source=SectionSource.BOOKMARK,
                heading_block_id=None,
                block_ids=(2,),
                start_page=2,
                end_page=2,
            ),
        ),
        chunks=(
            Chunk(
                position=0,
                section_position=0,
                kind=ChunkKind.PROSE,
                text="First sentence. Second sentence.",
                prefix_chars=0,
                token_count=6,
                start_page=1,
                end_page=1,
                spans=(SourceSpan(block_id=1, start=0, end=32),),
            ),
            Chunk(
                position=1,
                section_position=1,
                kind=ChunkKind.TABLE,
                text="| A | B |\n| 1 | 2 |",
                prefix_chars=0,
                token_count=10,
                start_page=2,
                end_page=2,
                spans=(SourceSpan(block_id=2, start=0, end=19),),
            ),
        ),
        report={"blocks": 3, "tables": 1},
    )


def helper_claimed(document_id: int, run: int) -> ClaimedDocument:
    return ClaimedDocument(document_id=document_id, run=run, media_type=MediaType.PDF, content=b"%PDF-1.7 report")


def helper_row_counts(database: Database) -> dict[str, int]:
    row = database.row(
        """SELECT (SELECT count(*) FROM document_sections) AS sections,
                  (SELECT count(*) FROM document_chunks) AS chunks""",
        (),
    )
    return {"sections": int(row["sections"]), "chunks": int(row["chunks"])} if row is not None else {}


def test___init__() -> None:
    database = Database("postgresql://u:p@h/d")
    sealer = SecretSealer(KEY)
    digest = TextDigest(KEY)
    tested = ProcessingStore(database, sealer, digest)
    result = (tested._database is database, tested._sealer is sealer, tested._digest is digest)
    expected = (True, True, True)
    assert result == expected


@pytest.mark.database
def test_claimed() -> None:
    database = helper_documents()
    tested = helper_store(database)
    # Taken once; a second claim, and a document that is not waiting, find nothing.
    result = (tested.claimed(1), tested.claimed(1), tested.claimed(2), tested.claimed(99))
    stored = database.row(
        "SELECT processing_status, processing_run, processing_started_at IS NOT NULL AS started FROM documents WHERE id = 1",
        (),
    )
    expected = (
        ClaimedDocument(document_id=1, run=1, media_type=MediaType.PDF, content=b"%PDF-1.7 report"),
        None,
        None,
        None,
    )
    assert result == expected
    exp_stored = {"processing_status": "processing", "processing_run": 1, "started": True}
    assert stored == exp_stored


@pytest.mark.database
def test_claimed__the_run_number_grows() -> None:
    database = helper_documents()
    tested = helper_store(database)
    tested.claimed(1)
    tested.requested(1, 1)
    result = tested.claimed(1)
    expected = ClaimedDocument(document_id=1, run=2, media_type=MediaType.PDF, content=b"%PDF-1.7 report")
    assert result == expected


@pytest.mark.database
def test_finished() -> None:
    database = helper_documents()
    tested = helper_store(database)
    claimed = tested.claimed(1)
    assert claimed is not None
    result = tested.finished(claimed, helper_result(), STRATEGY)
    document = database.row(
        """SELECT processing_status, processing_error, page_count, chunking_version, processed_at IS NOT NULL AS done,
               processing_report, extraction_sealed, index_status, index_error FROM documents WHERE id = 1""",
        (),
    )
    sections = database.rows(
        """SELECT id, document_id, parent_id, position, level, number, title_sealed, source, start_page, end_page,
               heading_block FROM document_sections ORDER BY position""",
        (),
    )
    chunks = database.rows(
        """SELECT document_id, section_id, position, kind, text_sealed, prefix_chars, token_count, start_page,
               end_page, source_spans, strategy, text_digest, embedding, embedding_model
               FROM document_chunks ORDER BY position""",
        (),
    )
    assert result is True
    assert document is not None
    sealer = SecretSealer(KEY)
    opened_document = {
        **{key: value for key, value in document.items() if key != "extraction_sealed"},
        "extraction": json.loads(sealer.opened(bytes(document["extraction_sealed"]))),
    }
    exp_document = {
        "processing_status": "completed",
        "processing_error": None,
        "page_count": 2,
        "chunking_version": STRATEGY,
        "done": True,
        "processing_report": {"blocks": 3, "tables": 1},
        "extraction": [block.to_dict() for block in helper_result().blocks],
        # The next stage waits for the new chunks.
        "index_status": "pending",
        "index_error": None,
    }
    assert opened_document == exp_document
    # Titles and texts are sealed: opened with the key, they are the whole originals.
    opened_sections = [{**row, "title_sealed": sealer.opened(bytes(row["title_sealed"]))} for row in sections]
    exp_sections = [
        {
            "id": 1,
            "document_id": 1,
            "parent_id": None,
            "position": 0,
            "level": 1,
            "number": "1",
            "title_sealed": "1. Intro",
            "source": "numbered",
            "start_page": 1,
            "end_page": 2,
            "heading_block": 0,
        },
        {
            "id": 2,
            "document_id": 1,
            "parent_id": 1,
            "position": 1,
            "level": 2,
            "number": "",
            "title_sealed": "Figures",
            "source": "bookmark",
            "start_page": 2,
            "end_page": 2,
            "heading_block": None,
        },
    ]
    assert opened_sections == exp_sections
    opened_chunks = [{**row, "text_sealed": sealer.opened(bytes(row["text_sealed"]))} for row in chunks]
    exp_chunks = [
        {
            "document_id": 1,
            "section_id": 1,
            "position": 0,
            "kind": "prose",
            "text_sealed": "First sentence. Second sentence.",
            "prefix_chars": 0,
            "token_count": 6,
            "start_page": 1,
            "end_page": 1,
            "source_spans": [{"block_id": 1, "start": 0, "end": 32}],
            "strategy": STRATEGY,
            "text_digest": DIGEST_PROSE,
            "embedding": None,
            "embedding_model": None,
        },
        {
            "document_id": 1,
            "section_id": 2,
            "position": 1,
            "kind": "table",
            "text_sealed": "| A | B |\n| 1 | 2 |",
            "prefix_chars": 0,
            "token_count": 10,
            "start_page": 2,
            "end_page": 2,
            "source_spans": [{"block_id": 2, "start": 0, "end": 19}],
            "strategy": STRATEGY,
            "text_digest": DIGEST_TABLE,
            "embedding": None,
            "embedding_model": None,
        },
    ]
    assert opened_chunks == exp_chunks


@pytest.mark.database
def test_finished__a_second_run_replaces_the_first() -> None:
    """Processing again never duplicates: the old sections and chunks go in
    the transaction that writes the new ones."""
    database = helper_documents()
    tested = helper_store(database)
    first = tested.claimed(1)
    assert first is not None
    tested.finished(first, helper_result(), STRATEGY)
    tested.requested(1, 1)
    second = tested.claimed(1)
    assert second is not None
    result = tested.finished(second, helper_result(), "other strategy")
    counts = helper_row_counts(database)
    strategies = database.rows("SELECT DISTINCT strategy FROM document_chunks", ())
    versions = database.row("SELECT chunking_version FROM documents WHERE id = 1", ())
    assert result is True
    exp_counts = {"sections": 2, "chunks": 2}
    assert counts == exp_counts
    exp_strategies = [{"strategy": "other strategy"}]
    assert strategies == exp_strategies
    exp_versions = {"chunking_version": "other strategy"}
    assert versions == exp_versions


@pytest.mark.database
def test_finished__an_unchanged_chunk_keeps_its_embedding() -> None:
    """Reprocessing sends only changed chunks to the indexing stage: a chunk
    whose passage is the same keeps its vector and model; a changed one has none
    (digest of "report › 1. Intro\\n\\nFirst sentence. Changed sentence.", from openssl)."""
    database = helper_documents()
    tested = helper_store(database)
    first = tested.claimed(1)
    assert first is not None
    tested.finished(first, helper_result(), STRATEGY)
    helper_embed(database, 0, 0, MODEL)
    helper_embed(database, 1, 1, MODEL)
    tested.requested(1, 1)
    second = tested.claimed(1)
    assert second is not None
    original = helper_result()
    changed = original._replace(
        chunks=(
            original.chunks[0]._replace(text="First sentence. Changed sentence."),
            original.chunks[1],
        )
    )
    result = tested.finished(second, changed, STRATEGY)
    stored = database.rows(
        "SELECT position, text_digest, embedding::text AS embedding, embedding_model FROM document_chunks ORDER BY position",
        (),
    )
    assert result is True
    exp_stored = [
        {
            "position": 0,
            "text_digest": "17d1485d8eaef1dd23003de19f23b9733a52a1e2d9c2f6d07574b084fdf9242a",
            "embedding": None,
            "embedding_model": None,
        },
        {"position": 1, "text_digest": DIGEST_TABLE, "embedding": helper_vector(1), "embedding_model": MODEL},
    ]
    assert stored == exp_stored


@pytest.mark.parametrize(
    "interruption",
    [
        pytest.param("UPDATE documents SET processing_status = 'pending' WHERE id = 1", id="reprocess_requested"),
        pytest.param("UPDATE documents SET processing_run = processing_run + 1 WHERE id = 1", id="a_newer_run"),
        pytest.param("DELETE FROM documents WHERE id = 1", id="document_deleted"),
    ],
)
@pytest.mark.database
def test_finished__a_run_that_lost_the_document_writes_nothing(interruption: str) -> None:
    database = helper_documents()
    tested = helper_store(database)
    claimed = tested.claimed(1)
    assert claimed is not None
    database.executed(interruption, ())
    result = tested.finished(claimed, helper_result(), STRATEGY)
    counts = helper_row_counts(database)
    assert result is False
    exp_counts = {"sections": 0, "chunks": 0}
    assert counts == exp_counts


@pytest.mark.database
def test_embeddings_of() -> None:
    """Only embedded chunks with a digest can be carried over."""
    database = helper_documents()
    tested = helper_store(database)
    claimed = tested.claimed(1)
    assert claimed is not None
    tested.finished(claimed, helper_result(), STRATEGY)
    helper_embed(database, 1, 5, "other-model")
    result = (tested.embeddings_of(1), tested.embeddings_of(3))
    expected = ({DIGEST_TABLE: EmbeddingCarry(vector=helper_vector(5), model="other-model")}, {})
    assert result == expected


@pytest.mark.database
def test_sections_written() -> None:
    database = helper_documents()
    tested = helper_store(database)
    result = tested.sections_written(3, helper_result())
    parents = database.rows("SELECT id, parent_id, document_id FROM document_sections ORDER BY id", ())
    expected = {0: 1, 1: 2}
    assert result == expected
    exp_parents = [
        {"id": 1, "parent_id": None, "document_id": 3},
        {"id": 2, "parent_id": 1, "document_id": 3},
    ]
    assert parents == exp_parents


@pytest.mark.database
def test_chunks_written() -> None:
    database = helper_documents()
    tested = helper_store(database)
    section_ids = tested.sections_written(3, helper_result())
    carried = {DIGEST_TABLE: EmbeddingCarry(vector=helper_vector(2), model=MODEL)}
    result = tested.chunks_written(3, "report.pdf", helper_result(), section_ids, STRATEGY, carried)
    stored = database.rows(
        """SELECT document_id, section_id, position, kind, strategy, text_digest, embedding::text AS embedding,
               embedding_model FROM document_chunks ORDER BY position""",
        (),
    )
    assert result is None
    exp_stored = [
        {
            "document_id": 3,
            "section_id": 1,
            "position": 0,
            "kind": "prose",
            "strategy": STRATEGY,
            "text_digest": DIGEST_PROSE,
            "embedding": None,
            "embedding_model": None,
        },
        {
            "document_id": 3,
            "section_id": 2,
            "position": 1,
            "kind": "table",
            "strategy": STRATEGY,
            "text_digest": DIGEST_TABLE,
            "embedding": helper_vector(2),
            "embedding_model": MODEL,
        },
    ]
    assert stored == exp_stored


@pytest.mark.database
def test_chunks_written__another_context_is_embedded_again() -> None:
    """The same text under another name or section path is another passage:
    its old embedding was made for the old context, and is not carried."""
    database = helper_documents()
    tested = helper_store(database)
    section_ids = tested.sections_written(3, helper_result())
    carried = {DIGEST_TABLE: EmbeddingCarry(vector=helper_vector(2), model=MODEL)}
    tested.chunks_written(3, "renamed.pdf", helper_result(), section_ids, STRATEGY, carried)
    result = database.rows("SELECT embedding, embedding_model FROM document_chunks ORDER BY position", ())
    expected = [{"embedding": None, "embedding_model": None}, {"embedding": None, "embedding_model": None}]
    assert result == expected


@pytest.mark.database
def test_closed() -> None:
    """A run ending without results clears the previous ones: nothing stale stays searchable."""
    database = helper_documents()
    tested = helper_store(database)
    first = tested.claimed(1)
    assert first is not None
    tested.finished(first, helper_result(), STRATEGY)
    tested.requested(1, 1)
    second = tested.claimed(1)
    assert second is not None
    result = tested.closed(second, ProcessingStatus.NO_TEXT, "No text.", {"pages_without_text": [1, 2]})
    document = database.row(
        """SELECT processing_status, processing_error, page_count, chunking_version, processing_report,
               extraction_sealed, processed_at IS NOT NULL AS done, index_status, index_error
           FROM documents WHERE id = 1""",
        (),
    )
    counts = helper_row_counts(database)
    assert result is True
    exp_document = {
        "processing_status": "no_text",
        "processing_error": "No text.",
        "page_count": None,
        "chunking_version": None,
        "processing_report": {"pages_without_text": [1, 2]},
        "extraction_sealed": None,
        "done": True,
        # No chunks: nothing for the indexing stage.
        "index_status": "not_indexed",
        "index_error": None,
    }
    assert document == exp_document
    exp_counts = {"sections": 0, "chunks": 0}
    assert counts == exp_counts


@pytest.mark.database
def test_closed__a_run_that_lost_the_document_changes_nothing() -> None:
    database = helper_documents()
    tested = helper_store(database)
    claimed = tested.claimed(1)
    assert claimed is not None
    tested.requested(1, 1)
    result = tested.closed(claimed, ProcessingStatus.FAILED, "Broken.", {})
    document = database.row("SELECT processing_status, processing_error FROM documents WHERE id = 1", ())
    assert result is False
    exp_document = {"processing_status": "pending", "processing_error": None}
    assert document == exp_document


@pytest.mark.database
def test_requested() -> None:
    """Only the owner's PDF goes back to the queue."""
    database = helper_documents()
    tested = helper_store(database)
    database.executed("UPDATE documents SET processing_status = 'completed'", ())
    result = (tested.requested(1, 1), tested.requested(1, 2), tested.requested(1, 3), tested.requested(1, 99))
    statuses = database.rows("SELECT id, processing_status FROM documents ORDER BY id", ())
    expected = (1, 0, 0, 0)
    assert result == expected
    exp_statuses = [
        {"id": 1, "processing_status": "pending"},
        {"id": 2, "processing_status": "completed"},
        {"id": 3, "processing_status": "completed"},
    ]
    assert statuses == exp_statuses


@pytest.mark.database
def test_pending_ids() -> None:
    database = helper_documents()
    tested = helper_store(database)
    tested.claimed(1)
    result = tested.pending_ids()
    expected = [3]
    assert result == expected


@pytest.mark.database
def test_restarted() -> None:
    """Only a run started long ago is put back: a young one belongs to a live process."""
    database = helper_documents()
    tested = helper_store(database)
    tested.claimed(1)
    tested.claimed(3)
    database.executed("UPDATE documents SET processing_started_at = now() - interval '11 minutes' WHERE id = %s", (1,))
    result = tested.restarted(10)
    statuses = database.rows("SELECT id, processing_status FROM documents ORDER BY id", ())
    expected = 1
    assert result == expected
    exp_statuses = [
        {"id": 1, "processing_status": "pending"},
        {"id": 2, "processing_status": "unsupported"},
        {"id": 3, "processing_status": "processing"},
    ]
    assert statuses == exp_statuses


@pytest.mark.database
def test_processing_of() -> None:
    database = helper_documents()
    tested = helper_store(database)
    claimed = tested.claimed(1)
    assert claimed is not None
    tested.finished(claimed, helper_result(), STRATEGY)
    helper_embed(database, 0, 0, MODEL)
    database.executed(
        """UPDATE documents SET processed_at = '2026-10-19 12:00:00+00', index_status = 'failed', index_model = %s,
               index_error = 'Broken.', index_report = '{"chunks": 2}', indexed_at = '2026-10-19 13:00:00+00'
           WHERE id = 1""",
        (MODEL,),
    )
    result = (tested.processing_of(1, 1), tested.processing_of(1, 2), tested.processing_of(1, 3))
    expected = (
        DocumentProcessing(
            document_id=1,
            name="report.pdf",
            media_type=MediaType.PDF,
            status=ProcessingStatus.COMPLETED,
            error="",
            page_count=2,
            chunking_version=STRATEGY,
            processed_at="2026-10-19T12:00:00Z",
            report={"blocks": 3, "tables": 1},
            index_status=IndexStatus.FAILED,
            index_model=MODEL,
            index_error="Broken.",
            indexed_at="2026-10-19T13:00:00Z",
            index_report={"chunks": 2},
            chunk_count=2,
            indexed_count=1,
        ),
        DocumentProcessing(
            document_id=2,
            name="photo.png",
            media_type=MediaType.PNG,
            status=ProcessingStatus.UNSUPPORTED,
            error="",
            page_count=None,
            chunking_version="",
            processed_at="",
            report={},
            index_status=IndexStatus.NOT_INDEXED,
            index_model="",
            index_error="",
            indexed_at="",
            index_report={},
            chunk_count=0,
            indexed_count=0,
        ),
        # Bob's document: not Ada's to see.
        None,
    )
    assert result == expected


@pytest.mark.database
def test_sections_of() -> None:
    database = helper_documents()
    tested = helper_store(database)
    claimed = tested.claimed(1)
    assert claimed is not None
    tested.finished(claimed, helper_result(), STRATEGY)
    result = (tested.sections_of(1), tested.sections_of(3))
    expected = (
        [
            StoredSection(
                section_id=1,
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
            ),
            StoredSection(
                section_id=2,
                parent_id=1,
                position=1,
                level=2,
                number="",
                title="Figures",
                source=SectionSource.BOOKMARK,
                start_page=2,
                end_page=2,
                heading_block_id=None,
                chunk_count=1,
            ),
        ],
        [],
    )
    assert result == expected


@pytest.mark.database
def test_chunks_of() -> None:
    database = helper_documents()
    tested = helper_store(database)
    claimed = tested.claimed(1)
    assert claimed is not None
    tested.finished(claimed, helper_result(), STRATEGY)
    helper_embed(database, 1, 3, MODEL)
    result = (tested.chunks_of(1), tested.chunks_of(3))
    expected = (
        [
            StoredChunk(
                chunk_id=1,
                section_id=1,
                position=0,
                kind=ChunkKind.PROSE,
                text="First sentence. Second sentence.",
                prefix_chars=0,
                token_count=6,
                start_page=1,
                end_page=1,
                spans=(SourceSpan(block_id=1, start=0, end=32),),
                strategy=STRATEGY,
                embedding_model="",
            ),
            StoredChunk(
                chunk_id=2,
                section_id=2,
                position=1,
                kind=ChunkKind.TABLE,
                text="| A | B |\n| 1 | 2 |",
                prefix_chars=0,
                token_count=10,
                start_page=2,
                end_page=2,
                spans=(SourceSpan(block_id=2, start=0, end=19),),
                strategy=STRATEGY,
                embedding_model=MODEL,
            ),
        ],
        [],
    )
    assert result == expected


@pytest.mark.database
def test_extraction_of() -> None:
    database = helper_documents()
    tested = helper_store(database)
    claimed = tested.claimed(1)
    assert claimed is not None
    tested.finished(claimed, helper_result(), STRATEGY)
    # Processed and Ada's; not processed; Bob's; unknown.
    result = (
        tested.extraction_of(1, 1),
        tested.extraction_of(1, 2),
        tested.extraction_of(1, 3),
        tested.extraction_of(1, 99),
    )
    expected = (list(helper_result().blocks), [], [], [])
    assert result == expected


@pytest.mark.database
def test_deleting_a_document_deletes_its_sections_and_chunks() -> None:
    database = helper_documents()
    tested = helper_store(database)
    claimed = tested.claimed(1)
    assert claimed is not None
    tested.finished(claimed, helper_result(), STRATEGY)
    before = helper_row_counts(database)
    DocumentStore(database, SecretSealer(KEY)).deleted(1, 1)
    result = (before, helper_row_counts(database))
    expected = ({"sections": 2, "chunks": 2}, {"sections": 0, "chunks": 0})
    assert result == expected
