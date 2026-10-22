from __future__ import annotations

import hashlib
import logging
import os
import threading
from unittest.mock import call, patch

import psycopg
import pytest

from rag_my_docs.libraries.database import Database
from rag_my_docs.structures.app_exception import AppException
from tests.conftest import helper_database


def helper_scratch() -> Database:
    """A Database of the test's own on the worker's scratch database, for the
    tests about connections and migrations: the shared one from
    `helper_database` is never closed or dropped under the other tests."""
    return Database(os.environ["TEST_DATABASE_URL"])


def helper_schema_dropped(database: Database) -> None:
    with database.transaction() as connection:
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")


def test___init__() -> None:
    tested = Database("postgresql://u:p@h/d")
    result = (tested._url, tested._opened, getattr(tested._local, "connection", None))
    expected = ("postgresql://u:p@h/d", [], None)
    assert result == expected


def test_migrations() -> None:
    tested = Database
    result = tested.migrations()
    expected = (
        (1, "accounts, sign-in, passkeys and documents", 9, "f007f8363455ca98a4bde55ea2b34768"),
        (2, "document processing: status, sections and chunks", 14, "9947fb59f22edbf13a7c0dec177390d7"),
        (3, "embeddings and indexing status", 12, "94e9dae1a17aea351110c849bed6d281"),
        (4, "hashed search terms and the evaluation set", 3, "87ae646ca8fe54ad432ba58994639acc"),
        (5, "language model connections and their steps", 2, "a9f5a27dbebb55794e9e3a8fe550f433"),
        (6, "drop what the demo no longer uses", 8, "529424bc9d43db836498ae0d6f874415"),
        (7, "drop the keyword model column", 1, "65e0fbd968d50e0914e058e604cca78b"),
    )
    text = "\n\n".join("\n".join(migration.statements) for migration in result)
    digests = tuple(
        (
            migration.version,
            migration.description,
            len(migration.statements),
            hashlib.md5("\n".join(migration.statements).encode()).hexdigest(),
        )
        for migration in result
    )
    assert digests == expected, text


@pytest.mark.database
def test_transaction() -> None:
    tested = helper_database()
    with tested.transaction() as outer:
        outer.execute("INSERT INTO schema_migrations (version, description) VALUES (%s, %s)", (50, "kept"))
        with tested.transaction() as inner:
            is_same = inner is outer
    with pytest.raises(AppException) as exc_info:
        with tested.transaction() as connection:
            connection.execute("INSERT INTO schema_migrations (version, description) VALUES (%s, %s)", (51, "x"))
            raise AppException(409, "Rolled back.")
    result = tested.rows("SELECT version FROM schema_migrations WHERE version >= %s ORDER BY version", (50,))
    tested.executed("DELETE FROM schema_migrations WHERE version >= %s", (50,))
    expected = [{"version": 50}]
    assert result == expected
    assert is_same is True
    exp_error = (409, "Rolled back.")
    assert (exc_info.value.status_code, exc_info.value.message) == exp_error


@pytest.mark.database
def test_transaction__a_database_error_is_unavailable(caplog: pytest.LogCaptureFixture) -> None:
    tested = helper_database()
    with caplog.at_level(logging.WARNING, logger="rag_my_docs"):
        with pytest.raises(AppException) as exc_info:
            tested.rows("SELECT * FROM no_such_table", ())
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (503, "The database is unavailable.")
    assert result == expected
    exp_messages = ["[DATABASE] UndefinedTable"]
    assert [record.getMessage() for record in caplog.records] == exp_messages


@pytest.mark.database
def test_rows() -> None:
    tested = helper_database()
    result = tested.rows("SELECT version, description FROM schema_migrations ORDER BY version", ())
    expected = [
        {"version": 1, "description": "accounts, sign-in, passkeys and documents"},
        {"version": 2, "description": "document processing: status, sections and chunks"},
        {"version": 3, "description": "embeddings and indexing status"},
        {"version": 4, "description": "hashed search terms and the evaluation set"},
        {"version": 5, "description": "language model connections and their steps"},
        {"version": 6, "description": "drop what the demo no longer uses"},
        {"version": 7, "description": "drop the keyword model column"},
    ]
    assert result == expected


@pytest.mark.database
def test_row() -> None:
    tested = helper_database()
    result = (
        tested.row("SELECT description FROM schema_migrations WHERE version = %s", (1,)),
        tested.row("SELECT description FROM schema_migrations WHERE version = %s", (999,)),
    )
    expected = ({"description": "accounts, sign-in, passkeys and documents"}, None)
    assert result == expected


@pytest.mark.database
def test_returned() -> None:
    tested = helper_database()
    result = tested.returned(
        "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s) RETURNING id, display_name",
        (b"sealed", "hash-ada", "Ada"),
    )
    expected = {"id": 1, "display_name": "Ada"}
    assert result == expected


def test_returned__nothing_returned_is_an_error() -> None:
    tested = Database("postgresql://u:p@h/d")
    with patch.object(Database, "row", side_effect=[None]) as mock_row:
        with pytest.raises(AppException) as exc_info:
            tested.returned("INSERT INTO t (a) VALUES (%s) RETURNING a", (1,))
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (500, "The database returned nothing.")
    assert result == expected
    exp_calls = [call("INSERT INTO t (a) VALUES (%s) RETURNING a", (1,))]
    assert mock_row.mock_calls == exp_calls


@pytest.mark.database
def test_executed() -> None:
    tested = helper_database()
    tested.executed(
        "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)", (b"s", "hash-ada", "Ada")
    )
    result = (
        tested.executed("UPDATE users SET display_name = %s WHERE id = %s", ("Ada L.", 1)),
        tested.executed("UPDATE users SET display_name = %s WHERE id = %s", ("Nobody", 999)),
    )
    expected = (1, 0)
    assert result == expected


@pytest.mark.database
def test_is_reachable() -> None:
    tested = helper_scratch()
    try:
        result = tested.is_reachable()
    finally:
        tested.closed()
    assert result is True


def test_is_reachable__a_dropped_connection_is_replaced_once() -> None:
    tested = Database("postgresql://u:p@h/d")
    with (
        patch.object(Database, "_is_answering", side_effect=[AppException(503, "x"), True]) as mock_answering,
        patch.object(Database, "_forget_connection", side_effect=[None]) as mock_forget,
    ):
        result = tested.is_reachable()
    assert result is True
    exp_calls = [call(), call()]
    assert mock_answering.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_forget.mock_calls == exp_calls


def test_is_reachable__unreachable() -> None:
    tested = Database("postgresql://u:p@h/d")
    with (
        patch.object(
            Database, "_is_answering", side_effect=[AppException(503, "x"), AppException(503, "x")]
        ) as mock_answering,
        patch.object(Database, "_forget_connection", side_effect=[None]) as mock_forget,
    ):
        result = tested.is_reachable()
    assert result is False
    exp_calls = [call(), call()]
    assert mock_answering.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_forget.mock_calls == exp_calls


@pytest.mark.database
def test_migrated() -> None:
    """Migration 1 is the first: no previous release exists whose queries
    could break on it. Migrations 2 to 7 have their own tests below, each
    running the previous release's writes on the migrated schema."""
    tested = helper_scratch()
    try:
        helper_schema_dropped(tested)
        result = (tested.migrated(), tested.migrated())
        ledger = tested.rows("SELECT version, description FROM schema_migrations ORDER BY version", ())
        tables = tested.rows(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename",
            (),
        )
    finally:
        tested.closed()
    expected = ((1, 2, 3, 4, 5, 6, 7), ())
    assert result == expected
    exp_ledger = [
        {"version": 1, "description": "accounts, sign-in, passkeys and documents"},
        {"version": 2, "description": "document processing: status, sections and chunks"},
        {"version": 3, "description": "embeddings and indexing status"},
        {"version": 4, "description": "hashed search terms and the evaluation set"},
        {"version": 5, "description": "language model connections and their steps"},
        {"version": 6, "description": "drop what the demo no longer uses"},
        {"version": 7, "description": "drop the keyword model column"},
    ]
    assert ledger == exp_ledger
    exp_tables = [
        {"tablename": "document_chunks"},
        {"tablename": "document_sections"},
        {"tablename": "documents"},
        {"tablename": "llm_connections"},
        {"tablename": "llm_steps"},
        {"tablename": "magic_links"},
        {"tablename": "schema_migrations"},
        {"tablename": "sessions"},
        {"tablename": "users"},
    ]
    assert tables == exp_tables


@pytest.mark.database
def test_migrated__version_2_is_additive() -> None:
    """Blue and green share the database during a deploy: the previous
    release (migration 1 only) must still write and delete documents on the
    schema migration 2 leaves, and a document it already stored that is not a
    PDF is marked as one the pipeline does not process."""
    tested = helper_scratch()
    first = Database.migrations()[:1]
    second = Database.migrations()[:2]
    try:
        helper_schema_dropped(tested)
        with patch.object(Database, "migrations", side_effect=[first]):
            before = tested.migrated()
        tested.executed(
            "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)", (b"s", "h", "Ada")
        )
        # The previous release's INSERT, column for column.
        previous_insert = """INSERT INTO documents (user_id, name_sealed, media_type, size_bytes, content_sealed)
                             VALUES (%s, %s, %s, %s, %s)"""
        tested.executed(previous_insert, (1, b"n", "image/png", 4, b"c"))
        tested.executed(previous_insert, (1, b"n", "application/pdf", 4, b"c"))
        with patch.object(Database, "migrations", side_effect=[second]):
            after = tested.migrated()
        tested.executed(previous_insert, (1, b"n", "application/pdf", 4, b"c"))
        statuses = tested.rows("SELECT id, processing_status, processing_run FROM documents ORDER BY id", ())
        # The previous release's DELETE, on a document that has sections and chunks.
        tested.executed(
            """INSERT INTO document_sections (document_id, position, level, number, title_sealed, source,
                   start_page, end_page) VALUES (3, 0, 1, '', %s, 'fallback', 1, 1)""",
            (b"t",),
        )
        tested.executed(
            """INSERT INTO document_chunks (document_id, section_id, position, kind, text_sealed, prefix_chars,
                   token_count, start_page, end_page, source_spans, strategy)
               VALUES (3, 1, 0, 'prose', %s, 0, 1, 1, 1, '[]', 's')""",
            (b"x",),
        )
        deleted = tested.executed("DELETE FROM documents WHERE id = %s AND user_id = %s", (3, 1))
        left = tested.rows(
            """SELECT (SELECT count(*) FROM document_sections) AS sections,
                      (SELECT count(*) FROM document_chunks) AS chunks""",
            (),
        )
    finally:
        tested.closed()
    result = (before, after, deleted)
    expected = ((1,), (2,), 1)
    assert result == expected
    exp_statuses = [
        {"id": 1, "processing_status": "unsupported", "processing_run": 0},
        {"id": 2, "processing_status": "pending", "processing_run": 0},
        {"id": 3, "processing_status": "pending", "processing_run": 0},
    ]
    assert statuses == exp_statuses
    exp_left = [{"sections": 0, "chunks": 0}]
    assert left == exp_left


@pytest.mark.database
def test_migrated__version_3_is_additive() -> None:
    """The previous release (migrations 1 and 2) must still store, process
    and delete documents on the schema migration 3 leaves; a document it had
    already processed is queued for indexing, and the others are not."""
    tested = helper_scratch()
    second = Database.migrations()[:2]
    third = Database.migrations()[:3]
    try:
        helper_schema_dropped(tested)
        with patch.object(Database, "migrations", side_effect=[second]):
            before = tested.migrated()
        tested.executed(
            "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)", (b"s", "h", "Ada")
        )
        # The previous release's INSERT, column for column.
        previous_insert = """INSERT INTO documents (user_id, name_sealed, media_type, size_bytes, content_sealed,
                                 processing_status) VALUES (%s, %s, %s, %s, %s, %s)"""
        tested.executed(previous_insert, (1, b"n", "application/pdf", 4, b"c", "completed"))
        tested.executed(previous_insert, (1, b"n", "image/png", 4, b"c", "unsupported"))
        with patch.object(Database, "migrations", side_effect=[third]):
            after = tested.migrated()
        tested.executed(previous_insert, (1, b"n", "application/pdf", 4, b"c", "processing"))
        # The previous release's end of a processing run, then its sections and chunks.
        tested.executed(
            """UPDATE documents SET processing_status = %s, processing_error = NULL, page_count = %s,
                   chunking_version = %s, processed_at = now(), processing_report = %s, extraction_sealed = %s
               WHERE id = %s AND processing_run = %s AND processing_status = %s""",
            ("completed", 1, "s", "{}", b"e", 3, 0, "processing"),
        )
        tested.executed(
            """INSERT INTO document_sections (document_id, position, level, number, title_sealed, source,
                   start_page, end_page) VALUES (3, 0, 1, '', %s, 'fallback', 1, 1)""",
            (b"t",),
        )
        tested.executed(
            """INSERT INTO document_chunks (document_id, section_id, position, kind, text_sealed, prefix_chars,
                   token_count, start_page, end_page, source_spans, strategy)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (3, 1, 0, "prose", b"x", 0, 1, 1, 1, "[]", "s"),
        )
        statuses = tested.rows(
            "SELECT id, processing_status, index_status, index_run, index_model FROM documents ORDER BY id", ()
        )
        chunks = tested.rows("SELECT embedding, embedding_model, text_digest FROM document_chunks", ())
        # The previous release's DELETE, on a document that has sections and chunks.
        deleted = tested.executed("DELETE FROM documents WHERE id = %s AND user_id = %s", (3, 1))
        left = tested.rows("SELECT count(*) AS chunks FROM document_chunks", ())
    finally:
        tested.closed()
    result = (before, after, deleted)
    expected = ((1, 2), (3,), 1)
    assert result == expected
    exp_statuses = [
        {"id": 1, "processing_status": "completed", "index_status": "pending", "index_run": 0, "index_model": None},
        {
            "id": 2,
            "processing_status": "unsupported",
            "index_status": "not_indexed",
            "index_run": 0,
            "index_model": None,
        },
        {"id": 3, "processing_status": "completed", "index_status": "not_indexed", "index_run": 0, "index_model": None},
    ]
    assert statuses == exp_statuses
    exp_chunks = [{"embedding": None, "embedding_model": None, "text_digest": None}]
    assert chunks == exp_chunks
    exp_left = [{"chunks": 0}]
    assert left == exp_left


@pytest.mark.database
def test_migrated__version_4_is_additive() -> None:
    """The previous release (migrations 1 to 3) must still create accounts,
    store, embed and delete documents on the schema migration 4 leaves; a
    deleted document takes its hashed terms with it."""
    tested = helper_scratch()
    third = Database.migrations()[:3]
    fourth = Database.migrations()[:4]
    try:
        helper_schema_dropped(tested)
        with patch.object(Database, "migrations", side_effect=[third]):
            before = tested.migrated()
        with patch.object(Database, "migrations", side_effect=[fourth]):
            after = tested.migrated()
        # The previous release's writes, column for column.
        tested.executed(
            "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)", (b"s", "h", "Ada")
        )
        tested.executed(
            """INSERT INTO documents (user_id, name_sealed, media_type, size_bytes, content_sealed, processing_status)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (1, b"n", "application/pdf", 4, b"c", "completed"),
        )
        tested.executed(
            """INSERT INTO document_sections (document_id, position, level, number, title_sealed, source,
                   start_page, end_page) VALUES (1, 0, 1, '', %s, 'fallback', 1, 1)""",
            (b"t",),
        )
        tested.executed(
            """INSERT INTO document_chunks (document_id, section_id, position, kind, text_sealed, prefix_chars,
                   token_count, start_page, end_page, source_spans, strategy, text_digest, embedding, embedding_model)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)""",
            (1, 1, 0, "prose", b"x", 0, 1, 1, 1, "[]", "s", "d", None, None),
        )
        tested.executed(
            "UPDATE document_chunks SET embedding = %s::vector, embedding_model = %s WHERE id = %s AND document_id = %s",
            ("[" + ",".join(["0"] * 383 + ["1"]) + "]", "m", 1, 1),
        )
        # This release's term of that chunk, then the previous release's DELETE.
        tested.executed(
            "INSERT INTO chunk_terms (chunk_id, document_id, term_hash, term_count) VALUES (1, 1, 'h', 2)", ()
        )
        stored = tested.rows("SELECT evaluation_sealed FROM users", ())
        deleted = tested.executed("DELETE FROM documents WHERE id = %s AND user_id = %s", (1, 1))
        left = tested.rows("SELECT (SELECT count(*) FROM chunk_terms) AS terms", ())
    finally:
        tested.closed()
    result = (before, after, deleted)
    expected = ((1, 2, 3), (4,), 1)
    assert result == expected
    exp_stored = [{"evaluation_sealed": None}]
    assert stored == exp_stored
    exp_left = [{"terms": 0}]
    assert left == exp_left


@pytest.mark.database
def test_migrated__version_5_is_additive() -> None:
    """New tables only: the previous release (migrations 1 to 4) keeps
    creating accounts and documents; a deleted connection leaves its step
    empty rather than failing."""
    tested = helper_scratch()
    fourth = Database.migrations()[:4]
    fifth = Database.migrations()[:5]
    try:
        helper_schema_dropped(tested)
        with patch.object(Database, "migrations", side_effect=[fourth, fifth]):
            before = tested.migrated()
            after = tested.migrated()
        # The previous release's writes, column for column.
        tested.executed(
            "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)", (b"s", "h", "Ada")
        )
        tested.executed(
            """INSERT INTO documents (user_id, name_sealed, media_type, size_bytes, content_sealed, processing_status)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (1, b"n", "application/pdf", 4, b"c", "completed"),
        )
        tested.executed(
            """INSERT INTO llm_connections (name, kind, base_url, api_key_sealed, keyword_model, answer_model)
               VALUES ('Local', 'openai', 'https://llm.example.com/v1', %s, 'local-model', 'local-model')""",
            (b"k",),
        )
        tested.executed("INSERT INTO llm_steps (step, connection_id) VALUES ('answers', 1)", ())
        tested.executed("DELETE FROM llm_connections WHERE id = 1", ())
        left = tested.rows("SELECT step, connection_id FROM llm_steps", ())
        thinking = tested.rows(
            "SELECT column_default FROM information_schema.columns WHERE column_name = 'disable_thinking'", ()
        )
    finally:
        tested.closed()
    result = (before, after)
    expected = ((1, 2, 3, 4), (5,))
    assert result == expected
    exp_left = [{"step": "answers", "connection_id": None}]
    assert left == exp_left
    exp_thinking = [{"column_default": "true"}]
    assert thinking == exp_thinking


@pytest.mark.database
def test_migrated__version_6_drops_only_what_the_previous_release_no_longer_uses() -> None:
    """The previous release (migrations 1 to 5, the demo before this one)
    keeps writing its rows, `keyword_model` included; this release's insert
    without it takes the new default. The dropped tables and columns, and
    the keywords step, are gone."""
    tested = helper_scratch()
    fifth = Database.migrations()[:5]
    sixth = Database.migrations()[:6]
    try:
        helper_schema_dropped(tested)
        with patch.object(Database, "migrations", side_effect=[fifth]):
            before = tested.migrated()
        tested.executed(
            "INSERT INTO llm_connections (name, kind, base_url, api_key_sealed, keyword_model, answer_model) "
            "VALUES ('Old', 'openai', 'https://llm.example.com/v1', %s, 'm', 'm')",
            (b"k",),
        )
        tested.executed("INSERT INTO llm_steps (step, connection_id) VALUES ('keywords', 1), ('answers', 1)", ())
        with patch.object(Database, "migrations", side_effect=[sixth]):
            after = tested.migrated()
        # The previous release's writes, column for column.
        tested.executed(
            "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)", (b"s", "h", "Ada")
        )
        tested.executed(
            """INSERT INTO llm_connections (name, kind, base_url, api_key_sealed, keyword_model, answer_model,
                   disable_thinking)
               VALUES ('Local', 'openai', 'https://llm.example.com/v1', %s, 'local-model', 'local-model', TRUE)""",
            (b"k",),
        )
        # This release's write, without the column the next migration drops.
        tested.executed(
            """INSERT INTO llm_connections (name, kind, base_url, api_key_sealed, answer_model, disable_thinking)
               VALUES ('Claude', 'claude', '', %s, 'claude-opus-5-5', FALSE)""",
            (b"k",),
        )
        keyword_models = tested.rows("SELECT id, keyword_model FROM llm_connections ORDER BY id", ())
        steps = tested.rows("SELECT step, connection_id FROM llm_steps", ())
        tables = tested.rows(
            """SELECT tablename FROM pg_tables WHERE schemaname = 'public'
                 AND tablename IN ('passkeys', 'webauthn_challenges', 'chunk_terms')""",
            (),
        )
        user_columns = tested.rows(
            """SELECT column_name FROM information_schema.columns WHERE table_name = 'users'
                 AND column_name IN ('is_admin', 'max_file_bytes', 'evaluation_sealed')""",
            (),
        )
    finally:
        tested.closed()
    result = (before, after)
    expected = ((1, 2, 3, 4, 5), (6,))
    assert result == expected
    exp_keyword_models = [
        {"id": 1, "keyword_model": "m"},
        {"id": 2, "keyword_model": "local-model"},
        {"id": 3, "keyword_model": ""},
    ]
    assert keyword_models == exp_keyword_models
    exp_steps = [{"step": "answers", "connection_id": 1}]
    assert steps == exp_steps
    exp_gone: list[dict[str, object]] = []
    assert (tables, user_columns) == (exp_gone, exp_gone)


@pytest.mark.database
def test_migrated__version_7_drops_the_column_the_previous_release_no_longer_writes() -> None:
    """The previous release (migrations 1 to 6) writes connections without
    `keyword_model`: its insert and update still run once the column is gone."""
    tested = helper_scratch()
    sixth = Database.migrations()[:6]
    try:
        helper_schema_dropped(tested)
        with patch.object(Database, "migrations", side_effect=[sixth]):
            before = tested.migrated()
        after = tested.migrated()
        # The previous release's writes, column for column.
        tested.executed(
            """INSERT INTO llm_connections (name, kind, base_url, api_key_sealed, answer_model, disable_thinking)
               VALUES ('Claude', 'claude', '', %s, 'claude-opus-5-5', FALSE)""",
            (b"k",),
        )
        tested.executed(
            """UPDATE llm_connections SET name = %s, kind = %s, base_url = %s, api_key_sealed = %s,
                   answer_model = %s, disable_thinking = %s
               WHERE id = %s""",
            ("Claude 2", "claude", "", b"k", "claude-sonnet-5-5", False, 1),
        )
        stored = tested.rows("SELECT name, answer_model FROM llm_connections", ())
        columns = tested.rows(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'llm_connections' "
            "AND column_name = 'keyword_model'",
            (),
        )
    finally:
        tested.closed()
    result = (before, after)
    expected = ((1, 2, 3, 4, 5, 6), (7,))
    assert result == expected
    exp_stored = [{"name": "Claude 2", "answer_model": "claude-sonnet-5-5"}]
    assert stored == exp_stored
    exp_columns: list[dict[str, object]] = []
    assert columns == exp_columns


@pytest.mark.database
def test_closed() -> None:
    tested = helper_scratch()
    connection = tested._connection()
    tested.closed()
    result = (connection.closed, tested._opened)
    expected = (True, [])
    assert result == expected


@pytest.mark.database
def test__is_answering() -> None:
    tested = helper_scratch()
    try:
        result = tested._is_answering()
    finally:
        tested.closed()
    assert result is True


@pytest.mark.database
def test__connection() -> None:
    tested = helper_scratch()
    try:
        first = tested._connection()
        again = tested._connection()
        first.close()
        reopened = tested._connection()
        result = (again is first, reopened is first, reopened.closed, reopened.autocommit, len(tested._opened))
        timeout = tested.row("SHOW statement_timeout", ())
    finally:
        tested.closed()
    expected = (True, False, False, True, 2)
    assert result == expected
    exp_timeout = {"statement_timeout": "30s"}
    assert timeout == exp_timeout


def test__connection__unreachable() -> None:
    tested = Database("postgresql://u:p@h/d")
    with patch.object(psycopg, "connect", side_effect=[psycopg.OperationalError("connection refused")]) as mock_connect:
        with pytest.raises(psycopg.OperationalError) as exc_info:
            tested._connection()
    result = (f"{exc_info.value}", tested._opened)
    expected = ("connection refused", [])
    assert result == expected
    exp_calls = [
        call(
            "postgresql://u:p@h/d",
            autocommit=True,
            row_factory=psycopg.rows.dict_row,
            connect_timeout=5,
            options="-c statement_timeout=30000",
        )
    ]
    assert mock_connect.mock_calls == exp_calls


@pytest.mark.database
def test__connection__a_finished_thread_s_connection_is_closed() -> None:
    tested = helper_scratch()
    opened: list[psycopg.Connection[psycopg.rows.DictRow]] = []
    try:
        worker = threading.Thread(target=lambda: opened.append(tested._connection()))
        worker.start()
        worker.join()
        mine = tested._connection()
        result = (opened[0].closed, mine.closed, tested._opened)
    finally:
        tested.closed()
    expected = (True, False, [(threading.current_thread(), mine)])
    assert result == expected


@pytest.mark.database
def test__living() -> None:
    tested = helper_scratch()
    try:
        mine = tested._connection()
        ended = threading.Thread(target=lambda: None)
        ended.start()
        ended.join()
        orphan = psycopg.connect(os.environ["TEST_DATABASE_URL"])
        tested._opened.append((ended, orphan))
        result = (tested._living(), orphan.closed, mine.closed)
    finally:
        tested.closed()
    expected = ([(threading.current_thread(), mine)], True, False)
    assert result == expected


@pytest.mark.database
def test__forget_connection() -> None:
    tested = helper_scratch()
    tested._forget_connection()
    connection = tested._connection()
    tested._forget_connection()
    result = (connection.closed, tested._opened, tested._local.connection)
    expected = (True, [], None)
    assert result == expected
