from __future__ import annotations

from typing import Any

import pytest

from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.document_store import DocumentStore
from rag_my_docs.libraries.processing_store import ProcessingStore
from rag_my_docs.libraries.search_store import SearchStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.libraries.text_digest import TextDigest
from rag_my_docs.structures.chunk import Chunk
from rag_my_docs.structures.processing_result import ProcessingResult
from rag_my_docs.structures.search_hit import SearchHit
from rag_my_docs.structures.section import Section
from rag_my_docs.structures.source_span import SourceSpan
from tests.conftest import helper_database

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="
STRATEGY = "sentences-v1 target=200 max=350 min=75 overlap=1"
MODEL = "BAAI/bge-small-en-v1.5"
OTHER_MODEL = "intfloat/e5-small-v2"


def helper_section(position: int, parent: int | None, title: str) -> Section:
    return Section(
        position=position,
        parent_position=parent,
        level=1 if parent is None else 2,
        number="",
        title=title,
        source=SectionSource.FALLBACK,
        heading_block_id=None,
        block_ids=(),
        start_page=1,
        end_page=2,
    )


def helper_chunk(position: int, section: int, text: str) -> Chunk:
    return Chunk(
        position=position,
        section_position=section,
        kind=ChunkKind.PROSE,
        text=text,
        prefix_chars=0,
        token_count=3,
        start_page=1,
        end_page=position + 1,
        spans=(SourceSpan(block_id=0, start=0, end=len(text)),),
    )


def helper_vector(values: dict[int, float]) -> str:
    """A 384-number vector as pgvector reads it, zero but at the given positions."""
    return "[" + ",".join(f"{values.get(index, 0.0):g}" for index in range(384)) + "]"


def helper_documents() -> Database:
    """Ada (1) and Bob (2).
    Document 1, Ada's report: sections 1 "1. Intro", 2 "1.1 Scope" inside it,
    and 3 untitled; chunks 1-4. Document 2, Ada's notes: section 4, chunks 5-6.
    Document 3, Bob's: section 5, chunk 7.
    Against a question along axis 0, cosine similarity is 1 for chunks 1, 4
    and 7, 0.8 for chunk 5, 0.6 for chunk 2, 0 for chunks 3 and 6 — but
    chunk 4 is from another model and chunk 7 is Bob's."""
    result = helper_database()
    for name in ("ada", "bob"):
        result.executed(
            "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)",
            (b"sealed", f"hash-{name}", name),
        )
    sealer = SecretSealer(KEY)
    documents = DocumentStore(result, sealer)
    processing = ProcessingStore(result, sealer, TextDigest(KEY))
    layouts: tuple[tuple[int, str, tuple[Section, ...], tuple[Chunk, ...]], ...] = (
        (
            1,
            "report.pdf",
            (helper_section(0, None, "1. Intro"), helper_section(1, 0, "1.1 Scope"), helper_section(2, None, "")),
            (
                helper_chunk(0, 0, "Sales grew."),
                helper_chunk(1, 1, "Two regions."),
                helper_chunk(2, 2, "Costs rose."),
                helper_chunk(3, 2, "Old model."),
            ),
        ),
        (
            1,
            "notes.pdf",
            (helper_section(0, None, "Notes"),),
            (helper_chunk(0, 0, "North."), helper_chunk(1, 0, "South.")),
        ),
        (2, "bob.pdf", (helper_section(0, None, "Bob"),), (helper_chunk(0, 0, "Bob's sales."),)),
    )
    for document_id, (user_id, name, sections, chunks) in enumerate(layouts, start=1):
        documents.created(user_id, name, MediaType.PDF, b"%PDF-1.7", ProcessingStatus.PENDING)
        claimed = processing.claimed(document_id)
        assert claimed is not None
        processing.finished(
            claimed, ProcessingResult(page_count=2, blocks=(), sections=sections, chunks=chunks, report={}), STRATEGY
        )
    for chunk_id, values, model in (
        (1, {0: 1.0}, MODEL),
        (2, {0: 3.0, 1: 4.0}, MODEL),
        (3, {2: 1.0}, MODEL),
        (4, {0: 1.0}, OTHER_MODEL),
        (5, {0: 4.0, 1: 3.0}, MODEL),
        (6, {3: 1.0}, MODEL),
        (7, {0: 1.0}, MODEL),
    ):
        result.executed(
            "UPDATE document_chunks SET embedding = %s::vector, embedding_model = %s WHERE id = %s",
            (helper_vector(values), model, chunk_id),
        )
    return result


def helper_store(database: Database) -> SearchStore:
    return SearchStore(database, SecretSealer(KEY), SectionPaths(database, SecretSealer(KEY)))


def helper_question() -> list[float]:
    return [1.0 if index == 0 else 0.0 for index in range(384)]


def helper_hit(rank: int, chunk_id: int, similarity: float) -> SearchHit:
    """Chunk `chunk_id` of the layout in `helper_documents`."""
    sources: dict[int, tuple[int, int, str, int, str, str]] = {
        1: (0, 1, "report.pdf", 1, "1. Intro", "Sales grew."),
        2: (1, 1, "report.pdf", 2, "1. Intro › 1.1 Scope", "Two regions."),
        3: (2, 1, "report.pdf", 3, "Untitled section", "Costs rose."),
        5: (0, 2, "notes.pdf", 4, "Notes", "North."),
        6: (1, 2, "notes.pdf", 4, "Notes", "South."),
    }
    position, document_id, name, section_id, path, text = sources[chunk_id]
    return SearchHit(
        rank=rank,
        chunk_id=chunk_id,
        chunk_position=position,
        document_id=document_id,
        document_name=name,
        section_id=section_id,
        section_path=path,
        start_page=1,
        end_page=position + 1,
        text=text,
        prefix_chars=0,
        similarity=similarity,
    )


def helper_rounded(hits: list[SearchHit]) -> list[SearchHit]:
    """Similarities to six decimals: pgvector stores single-precision floats."""
    return [hit._replace(similarity=round(hit.similarity, 6)) for hit in hits]


def test___init__() -> None:
    database = Database("postgresql://u:p@h/d")
    sealer = SecretSealer(KEY)
    paths = SectionPaths(database, sealer)
    tested = SearchStore(database, sealer, paths)
    result = (tested._database is database, tested._sealer is sealer, tested._paths is paths)
    expected = (True, True, True)
    assert result == expected


@pytest.mark.database
def test_indexed_count() -> None:
    """Only the person's documents indexed by this model count."""
    database = helper_documents()
    tested = helper_store(database)
    before = tested.indexed_count(1, MODEL)
    database.executed("UPDATE documents SET index_status = 'indexed', index_model = %s WHERE id IN (1, 3)", (MODEL,))
    database.executed("UPDATE documents SET index_status = 'indexed', index_model = %s WHERE id = 2", (OTHER_MODEL,))
    result = (before, tested.indexed_count(1, MODEL), tested.indexed_count(2, MODEL), tested.indexed_count(1, "x"))
    expected = (0, 1, 1, 0)
    assert result == expected


@pytest.mark.database
def test_nearest() -> None:
    """Most similar first; ties keep document order; only the person's
    chunks, only this model's."""
    database = helper_documents()
    tested = helper_store(database)
    result = helper_rounded(tested.nearest(1, helper_question(), MODEL, 10))
    expected = [
        helper_hit(1, 1, 1.0),
        helper_hit(2, 5, 0.8),
        helper_hit(3, 2, 0.6),
        helper_hit(4, 3, 0.0),
        helper_hit(5, 6, 0.0),
    ]
    assert result == expected


@pytest.mark.parametrize(
    "top_k, expected",
    [
        pytest.param(2, [helper_hit(1, 1, 1.0), helper_hit(2, 5, 0.8)], id="two"),
        pytest.param(1, [helper_hit(1, 1, 1.0)], id="one"),
    ],
)
@pytest.mark.database
def test_nearest__limited(top_k: int, expected: list[SearchHit]) -> None:
    database = helper_documents()
    tested = helper_store(database)
    result = helper_rounded(tested.nearest(1, helper_question(), MODEL, top_k))
    assert result == expected


def test_hit() -> None:
    sealer = SecretSealer(KEY)
    tested = SearchStore(
        Database("postgresql://u:p@h/d"), sealer, SectionPaths(Database("postgresql://u:p@h/d"), sealer)
    )
    row: dict[str, Any] = {
        "id": "12",
        "position": 2,
        "document_id": 3,
        "name_sealed": memoryview(sealer.sealed("report.pdf")),
        "section_id": 7,
        "start_page": 2,
        "end_page": 3,
        "text_sealed": memoryview(sealer.sealed("Sales grew.")),
        "prefix_chars": 4,
        "similarity": 0.75,
    }
    result = tested.hit(1, row, {7: "2. Results"})
    expected = SearchHit(
        rank=1,
        chunk_id=12,
        chunk_position=2,
        document_id=3,
        document_name="report.pdf",
        section_id=7,
        section_path="2. Results",
        start_page=2,
        end_page=3,
        text="Sales grew.",
        prefix_chars=4,
        similarity=0.75,
    )
    assert result == expected
