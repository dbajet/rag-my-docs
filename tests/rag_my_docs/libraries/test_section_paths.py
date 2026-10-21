from __future__ import annotations

import pytest

from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.document_store import DocumentStore
from rag_my_docs.libraries.processing_store import ProcessingStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.libraries.text_digest import TextDigest
from rag_my_docs.structures.chunk import Chunk
from rag_my_docs.structures.processing_result import ProcessingResult
from rag_my_docs.structures.section import Section
from rag_my_docs.structures.source_span import SourceSpan
from tests.conftest import helper_database

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="


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
        end_page=1,
    )


def helper_documents() -> Database:
    """Ada's document 1 with sections 1 "1. Intro", 2 "1.1 Scope" inside it and
    3 untitled; document 2 with section 4 "Notes"."""
    result = helper_database()
    result.executed(
        "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)", (b"s", "hash-ada", "Ada")
    )
    sealer = SecretSealer(KEY)
    documents = DocumentStore(result, sealer)
    processing = ProcessingStore(result, sealer, TextDigest(KEY))
    chunk = Chunk(
        position=0,
        section_position=0,
        kind=ChunkKind.PROSE,
        text="Text.",
        prefix_chars=0,
        token_count=2,
        start_page=1,
        end_page=1,
        spans=(SourceSpan(block_id=0, start=0, end=5),),
    )
    for document_id, sections in (
        (1, (helper_section(0, None, "1. Intro"), helper_section(1, 0, "1.1 Scope"), helper_section(2, None, ""))),
        (2, (helper_section(0, None, "Notes"),)),
    ):
        documents.created(1, f"{document_id}.pdf", MediaType.PDF, b"%PDF-1.7", ProcessingStatus.PENDING)
        claimed = processing.claimed(document_id)
        assert claimed is not None
        processing.finished(
            claimed, ProcessingResult(page_count=1, blocks=(), sections=sections, chunks=(chunk,), report={}), "s"
        )
    return result


def test___init__() -> None:
    database = Database("postgresql://u:p@h/d")
    sealer = SecretSealer(KEY)
    tested = SectionPaths(database, sealer)
    result = (tested._database is database, tested._sealer is sealer)
    expected = (True, True)
    assert result == expected


@pytest.mark.database
def test_paths() -> None:
    database = helper_documents()
    tested = SectionPaths(database, SecretSealer(KEY))
    result = (tested.paths([1, 2]), tested.paths([2]), tested.paths([]))
    expected = (
        {1: "1. Intro", 2: "1. Intro › 1.1 Scope", 3: "Untitled section", 4: "Notes"},
        {4: "Notes"},
        {},
    )
    assert result == expected


@pytest.mark.parametrize(
    "titles, parents, expected",
    [
        pytest.param(
            {0: "2. Results", 1: "2.1 Details", 2: "2.1.1 Costs"},
            {0: None, 1: 0, 2: 1},
            {0: "2. Results", 1: "2. Results › 2.1 Details", 2: "2. Results › 2.1 Details › 2.1.1 Costs"},
            id="nested",
        ),
        pytest.param(
            {5: "", 6: "Annex"}, {5: None, 6: 5}, {5: "Untitled section", 6: "Untitled section › Annex"}, id="untitled"
        ),
        pytest.param({}, {}, {}, id="none"),
    ],
)
def test_joined(titles: dict[int, str], parents: dict[int, int | None], expected: dict[int, str]) -> None:
    tested = SectionPaths
    result = tested.joined(titles, parents)
    assert result == expected
