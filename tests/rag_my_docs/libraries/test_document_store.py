from __future__ import annotations

import pytest

from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.document_store import DocumentStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.structures.document import Document
from rag_my_docs.structures.document_file import DocumentFile
from tests.conftest import helper_database

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="


def helper_people() -> Database:
    """Two people, and a fixed upload moment so the time a document shows is
    known: 01:30 in Paris is 23:30 the day before in UTC. The changed default
    makes the next `helper_database` rebuild the schema."""
    result = helper_database()
    for name in ("ada", "bob"):
        result.executed(
            "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)",
            (b"sealed", f"hash-{name}", name),
        )
    result.executed("ALTER TABLE documents ALTER COLUMN created_at SET DEFAULT '2026-10-19 01:30:00+02'", ())
    return result


def helper_store(database: Database) -> DocumentStore:
    return DocumentStore(database, SecretSealer(KEY))


def helper_document(
    document_id: int,
    user_id: int,
    name: str,
    media_type: MediaType,
    size_bytes: int,
    status: ProcessingStatus,
) -> Document:
    return Document(
        document_id=document_id,
        user_id=user_id,
        name=name,
        media_type=media_type,
        size_bytes=size_bytes,
        created_at="2026-10-18T23:30:00Z",
        processing_status=status,
        index_status=IndexStatus.NOT_INDEXED,
        chunk_count=0,
        indexed_count=0,
    )


def helper_chunks(database: Database, document_id: int) -> None:
    """A section and three chunks for the document, one of them with an embedding."""
    database.executed(
        """INSERT INTO document_sections (document_id, position, level, number, title_sealed, source, start_page, end_page)
           VALUES (%s, 0, 1, '', %s, 'fallback', 1, 1)""",
        (document_id, b"sealed"),
    )
    for position in range(3):
        database.executed(
            """INSERT INTO document_chunks (document_id, section_id, position, kind, text_sealed, prefix_chars,
                   token_count, start_page, end_page, source_spans, strategy)
               VALUES (%s, (SELECT max(id) FROM document_sections), %s, 'prose', %s, 0, 1, 1, 1, '[]', 's')""",
            (document_id, position, b"sealed"),
        )
    vector = "[" + ",".join(["1"] + ["0"] * 383) + "]"
    database.executed(
        "UPDATE document_chunks SET embedding = %s::vector, embedding_model = 'm' WHERE document_id = %s AND position = 0",
        (vector, document_id),
    )


def test___init__() -> None:
    database = Database("postgresql://u:p@h/d")
    sealer = SecretSealer(KEY)
    tested = DocumentStore(database, sealer)
    result = (tested._database is database, tested._sealer is sealer)
    expected = (True, True)
    assert result == expected


@pytest.mark.database
def test_created() -> None:
    database = helper_people()
    tested = helper_store(database)
    result = tested.created(2, "Relevé été.pdf", MediaType.PDF, b"%PDF-1.7 content", ProcessingStatus.PENDING)
    stored = database.row(
        """SELECT user_id, name_sealed, media_type, size_bytes, content_sealed, processing_status
           FROM documents WHERE id = %s""",
        (1,),
    )
    expected = helper_document(1, 2, "Relevé été.pdf", MediaType.PDF, 16, ProcessingStatus.PENDING)
    assert result == expected
    # Sealed at rest: only the key opens the stored bytes back into the name and the content.
    sealer = SecretSealer(KEY)
    opened = (
        stored["user_id"],
        sealer.opened(bytes(stored["name_sealed"])),
        stored["media_type"],
        stored["size_bytes"],
        sealer.opened_bytes(bytes(stored["content_sealed"])),
        stored["processing_status"],
    )
    exp_opened = (2, "Relevé été.pdf", "application/pdf", 16, b"%PDF-1.7 content", "pending")
    assert opened == exp_opened


@pytest.mark.database
def test_of_user() -> None:
    database = helper_people()
    tested = helper_store(database)
    tested.created(1, "first.txt", MediaType.TEXT, b"one", ProcessingStatus.UNSUPPORTED)
    tested.created(2, "bob.png", MediaType.PNG, b"\x89PNG", ProcessingStatus.UNSUPPORTED)
    tested.created(1, "second.md", MediaType.MARKDOWN, b"# two", ProcessingStatus.UNSUPPORTED)
    tested.created(1, "older.csv", MediaType.CSV, b"a,b", ProcessingStatus.UNSUPPORTED)
    database.executed("UPDATE documents SET created_at = created_at - interval '1 day' WHERE id = %s", (4,))
    database.executed("UPDATE documents SET index_status = 'indexed' WHERE id = %s", (4,))
    helper_chunks(database, 4)
    result = (tested.of_user(1), tested.of_user(3))
    # In the order stored, the corpus's own: an older date does not move a document up.
    expected = (
        [
            helper_document(1, 1, "first.txt", MediaType.TEXT, 3, ProcessingStatus.UNSUPPORTED),
            helper_document(3, 1, "second.md", MediaType.MARKDOWN, 5, ProcessingStatus.UNSUPPORTED),
            Document(
                document_id=4,
                user_id=1,
                name="older.csv",
                media_type=MediaType.CSV,
                size_bytes=3,
                created_at="2026-10-17T23:30:00Z",
                processing_status=ProcessingStatus.UNSUPPORTED,
                index_status=IndexStatus.INDEXED,
                chunk_count=3,
                indexed_count=1,
            ),
        ],
        [],
    )
    assert result == expected


@pytest.mark.database
def test_file_of() -> None:
    database = helper_people()
    tested = helper_store(database)
    tested.created(1, "notes.txt", MediaType.TEXT, b"Ada's notes", ProcessingStatus.UNSUPPORTED)
    tested.created(2, "bob.gif", MediaType.GIF, b"GIF89a", ProcessingStatus.UNSUPPORTED)
    # Another person's document, like an unknown one, is not found.
    result = (tested.file_of(1, 1), tested.file_of(1, 2), tested.file_of(1, 99))
    expected = (DocumentFile(name="notes.txt", media_type=MediaType.TEXT, content=b"Ada's notes"), None, None)
    assert result == expected


@pytest.mark.database
def test_deleted() -> None:
    database = helper_people()
    tested = helper_store(database)
    tested.created(1, "notes.txt", MediaType.TEXT, b"Ada's notes", ProcessingStatus.UNSUPPORTED)
    tested.created(2, "bob.gif", MediaType.GIF, b"GIF89a", ProcessingStatus.UNSUPPORTED)
    # Someone else's document is not removed, and says so with a zero.
    result = (tested.deleted(1, 2), tested.deleted(1, 1), tested.deleted(1, 1))
    stored = database.rows("SELECT id, user_id FROM documents ORDER BY id", ())
    expected = (0, 1, 0)
    assert result == expected
    exp_stored = [{"id": 2, "user_id": 2}]
    assert stored == exp_stored


def test__document() -> None:
    tested = helper_store(Database("postgresql://u:p@h/d"))
    row = {
        "id": 7,
        "user_id": 2,
        "name_sealed": SecretSealer(KEY).sealed("scan.jpg"),
        "media_type": "image/jpeg",
        "size_bytes": 2048,
        "created_at": "2026-10-18T23:30:00Z",
        "processing_status": "unsupported",
        "index_status": "not_indexed",
        "chunk_count": 0,
        "indexed_count": 0,
    }
    result = tested._document(row)
    expected = helper_document(7, 2, "scan.jpg", MediaType.JPEG, 2048, ProcessingStatus.UNSUPPORTED)
    assert result == expected
