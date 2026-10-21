from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import DictRow, dict_row

from rag_my_docs.constants.constants import Constants
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.migration import Migration


class Database:
    """The schema, its migrations, and the connections that run them.

    Raw SQL through psycopg and nothing in between. Each thread of the
    request pool keeps one connection of its own. Migrations are
    numbered, handwritten and additive: blue and green share this database
    for the length of a deploy, so nothing is renamed or dropped in the
    release that stops using it."""

    _UNAVAILABLE = "The database is unavailable."
    _NOTHING_RETURNED = "The database returned nothing."

    _LEDGER = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )"""

    def __init__(self, url: str) -> None:
        self._url = url
        # The connection and the open transaction of each thread: requests are
        # served on a pool of threads, and each one's transaction is its own.
        self._local = threading.local()
        self._opened: list[psycopg.Connection[DictRow]] = []
        self._opened_lock = threading.Lock()

    @classmethod
    def migrations(cls) -> tuple[Migration, ...]:
        """One entry per change, ever, numbered from 1 without a gap. The
        tables arrive with the code that reads them."""
        return (
            Migration(
                version=1,
                description="accounts, sign-in, passkeys and documents",
                statements=(
                    # No password column, not even an empty one. The email
                    # is personal data: sealed, and found through its keyed blind
                    # index, so a dump of this table is not a mailing list.
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id             BIGSERIAL PRIMARY KEY,
                        email_sealed   BYTEA NOT NULL,
                        email_hash     TEXT NOT NULL UNIQUE,
                        display_name   TEXT NOT NULL,
                        is_admin       BOOLEAN NOT NULL DEFAULT FALSE,
                        max_file_bytes BIGINT NOT NULL DEFAULT 10485760,
                        created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                        disabled_at    TIMESTAMPTZ NULL
                    )""",
                    # A session is a token the browser holds and a hash the table
                    # holds: signing out deletes the row, and the cookie dies with it.
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id         BIGSERIAL PRIMARY KEY,
                        user_id    BIGINT NOT NULL REFERENCES users(id),
                        token_hash TEXT NOT NULL UNIQUE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        expires_at TIMESTAMPTZ NOT NULL
                    )""",
                    "CREATE INDEX IF NOT EXISTS sessions_user ON sessions (user_id)",
                    # A sign-in link or an invitation: single use, consumed by one
                    # atomic UPDATE so two simultaneous clicks cannot both win.
                    """
                    CREATE TABLE IF NOT EXISTS magic_links (
                        id         BIGSERIAL PRIMARY KEY,
                        user_id    BIGINT NOT NULL REFERENCES users(id),
                        token_hash TEXT NOT NULL UNIQUE,
                        purpose    TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        expires_at TIMESTAMPTZ NOT NULL,
                        used_at    TIMESTAMPTZ NULL
                    )""",
                    """
                    CREATE TABLE IF NOT EXISTS passkeys (
                        id            BIGSERIAL PRIMARY KEY,
                        user_id       BIGINT NOT NULL REFERENCES users(id),
                        credential_id BYTEA NOT NULL UNIQUE,
                        public_key    BYTEA NOT NULL,
                        sign_count    BIGINT NOT NULL,
                        transports    TEXT NOT NULL,
                        label         TEXT NOT NULL,
                        created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                        last_used_at  TIMESTAMPTZ NULL
                    )""",
                    "CREATE INDEX IF NOT EXISTS passkeys_user ON passkeys (user_id)",
                    # A WebAuthn challenge, single use and short-lived. A sign-in
                    # challenge names nobody: the passkey says whose it is.
                    """
                    CREATE TABLE IF NOT EXISTS webauthn_challenges (
                        id         BIGSERIAL PRIMARY KEY,
                        challenge  BYTEA NOT NULL,
                        purpose    TEXT NOT NULL,
                        user_id    BIGINT NULL REFERENCES users(id),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        expires_at TIMESTAMPTZ NOT NULL,
                        used_at    TIMESTAMPTZ NULL
                    )""",
                    # The documents live in the database so the daily sealed dump
                    # holds them too, and they are sealed themselves, name and
                    # content: a dump without the data key shows sizes and types,
                    # nothing a person wrote.
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id             BIGSERIAL PRIMARY KEY,
                        user_id        BIGINT NOT NULL REFERENCES users(id),
                        name_sealed    BYTEA NOT NULL,
                        media_type     TEXT NOT NULL,
                        size_bytes     BIGINT NOT NULL,
                        content_sealed BYTEA NOT NULL,
                        created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                    )""",
                    "CREATE INDEX IF NOT EXISTS documents_user ON documents (user_id, created_at DESC)",
                ),
            ),
            Migration(
                version=2,
                description="document processing: status, sections and chunks",
                statements=(
                    # Every new column is nullable or has a default, so the previous
                    # release's INSERT INTO documents still works during a deploy.
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS processing_status TEXT NOT NULL DEFAULT 'pending'",
                    # Each attempt takes the next number; only the attempt that still
                    # holds the document may store its results.
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS processing_run INTEGER NOT NULL DEFAULT 0",
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS processing_started_at TIMESTAMPTZ NULL",
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ NULL",
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS processing_error TEXT NULL",
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_count INTEGER NULL",
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunking_version TEXT NULL",
                    # Counts and page numbers only, never document text: stored unsealed.
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS processing_report JSONB NULL",
                    # The extracted blocks, for the inspection view: document text, sealed.
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS extraction_sealed BYTEA NULL",
                    "UPDATE documents SET processing_status = 'unsupported' WHERE media_type <> 'application/pdf'",
                    # A section's title and a chunk's text are document text: sealed.
                    # Deleting a document deletes its sections, which delete their chunks.
                    """
                    CREATE TABLE IF NOT EXISTS document_sections (
                        id              BIGSERIAL PRIMARY KEY,
                        document_id     BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        parent_id       BIGINT NULL REFERENCES document_sections(id) ON DELETE CASCADE,
                        position        INTEGER NOT NULL,
                        level           INTEGER NOT NULL,
                        number          TEXT NOT NULL,
                        title_sealed    BYTEA NOT NULL,
                        source          TEXT NOT NULL,
                        start_page      INTEGER NOT NULL,
                        end_page        INTEGER NOT NULL,
                        heading_block   INTEGER NULL,
                        UNIQUE (document_id, position)
                    )""",
                    """
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id           BIGSERIAL PRIMARY KEY,
                        document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        section_id   BIGINT NOT NULL REFERENCES document_sections(id) ON DELETE CASCADE,
                        position     INTEGER NOT NULL,
                        kind         TEXT NOT NULL,
                        text_sealed  BYTEA NOT NULL,
                        prefix_chars INTEGER NOT NULL,
                        token_count  INTEGER NOT NULL,
                        start_page   INTEGER NOT NULL,
                        end_page     INTEGER NOT NULL,
                        source_spans JSONB NOT NULL,
                        strategy     TEXT NOT NULL,
                        UNIQUE (document_id, position)
                    )""",
                    "CREATE INDEX IF NOT EXISTS document_chunks_section ON document_chunks (section_id)",
                    "CREATE INDEX IF NOT EXISTS documents_processing ON documents (processing_status)",
                ),
            ),
            Migration(
                version=3,
                description="embeddings and indexing status",
                statements=(
                    "CREATE EXTENSION IF NOT EXISTS vector",
                    # One embedding per chunk, from the model named beside it. Its size
                    # is the model's (384 for bge-small): written out, since a
                    # migration is frozen SQL that must keep meaning what it meant.
                    # Exact search over every row: no approximate index for so few.
                    "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector(384) NULL",
                    "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT NULL",
                    # A keyed fingerprint of the chunk's text, never the text: an
                    # unchanged chunk keeps its embedding when the document is processed again.
                    "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS text_digest TEXT NULL",
                    # The indexing stage, apart from extraction: nullable or defaulted,
                    # so the previous release's writes still work during a deploy.
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS index_status TEXT NOT NULL DEFAULT 'not_indexed'",
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS index_run INTEGER NOT NULL DEFAULT 0",
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS index_started_at TIMESTAMPTZ NULL",
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMPTZ NULL",
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS index_model TEXT NULL",
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS index_error TEXT NULL",
                    # Counts and durations only, never text.
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS index_report JSONB NULL",
                    # Documents already processed are indexed at the next start.
                    "UPDATE documents SET index_status = 'pending' WHERE processing_status = 'completed'",
                ),
            ),
            Migration(
                version=4,
                description="hashed search terms and the evaluation set",
                statements=(
                    # Which keyed-hash words each chunk holds, for matching keywords
                    # word for word against encrypted text; gone with their chunk.
                    """
                    CREATE TABLE IF NOT EXISTS chunk_terms (
                        chunk_id    BIGINT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
                        document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        term_hash   TEXT NOT NULL,
                        term_count  INTEGER NOT NULL,
                        PRIMARY KEY (chunk_id, term_hash)
                    )
                    """,
                    "CREATE INDEX IF NOT EXISTS chunk_terms_term ON chunk_terms (term_hash, document_id)",
                    # A person's test questions and expected facts, sealed: they say
                    # what the documents contain.
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS evaluation_sealed BYTEA NULL",
                ),
            ),
            Migration(
                version=5,
                description="language model connections and their steps",
                statements=(
                    # Set up by admins in Settings; the key sealed like every secret at rest.
                    """
                    CREATE TABLE IF NOT EXISTS llm_connections (
                        id               BIGSERIAL PRIMARY KEY,
                        name             TEXT NOT NULL,
                        kind             TEXT NOT NULL,
                        base_url         TEXT NOT NULL,
                        api_key_sealed   BYTEA NOT NULL,
                        keyword_model    TEXT NOT NULL,
                        answer_model     TEXT NOT NULL,
                        disable_thinking BOOLEAN NOT NULL DEFAULT TRUE,
                        created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """,
                    # Which connection does each step; a deleted connection leaves its step empty.
                    """
                    CREATE TABLE IF NOT EXISTS llm_steps (
                        step          TEXT PRIMARY KEY,
                        connection_id BIGINT NULL REFERENCES llm_connections(id) ON DELETE SET NULL
                    )
                    """,
                ),
            ),
            # The contract half of expand-then-contract: what the previous
            # release already stopped using. `keyword_model`, which it still
            # writes, only gets a default here, so this release can leave it
            # out; it goes in the next migration.
            Migration(
                version=6,
                description="drop what the demo no longer uses",
                statements=(
                    "DROP TABLE IF EXISTS passkeys",
                    "DROP TABLE IF EXISTS webauthn_challenges",
                    "DROP TABLE IF EXISTS chunk_terms",
                    "ALTER TABLE users DROP COLUMN IF EXISTS is_admin",
                    "ALTER TABLE users DROP COLUMN IF EXISTS max_file_bytes",
                    "ALTER TABLE users DROP COLUMN IF EXISTS evaluation_sealed",
                    "DELETE FROM llm_steps WHERE step = 'keywords'",
                    "ALTER TABLE llm_connections ALTER COLUMN keyword_model SET DEFAULT ''",
                ),
            ),
            # The previous release no longer writes it (migration 6 gave it a default).
            Migration(
                version=7,
                description="drop the keyword model column",
                statements=("ALTER TABLE llm_connections DROP COLUMN IF EXISTS keyword_model",),
            ),
        )

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection[DictRow]]:
        """One transaction: committed when the block ends, rolled back when it
        raises. A transaction opened inside another on the same thread is the
        outer one, so a command can make several stores' writes atomic."""
        if (current := getattr(self._local, "transaction", None)) is not None:
            yield current
            return
        try:
            connection = self._connection()
            with connection.transaction():
                self._local.transaction = connection
                try:
                    yield connection
                finally:
                    self._local.transaction = None
        except psycopg.Error as error:
            logging.getLogger(Constants.logger_name).warning("[DATABASE] %s", type(error).__name__)
            raise AppException(503, self._UNAVAILABLE) from error

    def rows(self, sql: str, parameters: tuple[Any, ...]) -> list[dict[str, Any]]:
        with self.transaction() as connection:
            return [dict(row) for row in connection.execute(sql, parameters).fetchall()]

    def row(self, sql: str, parameters: tuple[Any, ...]) -> dict[str, Any] | None:
        with self.transaction() as connection:
            found = connection.execute(sql, parameters).fetchone()
            return dict(found) if found is not None else None

    def returned(self, sql: str, parameters: tuple[Any, ...]) -> dict[str, Any]:
        """The row an `INSERT … RETURNING` always returns: its absence is a
        broken statement, not a case to handle."""
        result = self.row(sql, parameters)
        if result is None:
            raise AppException(500, self._NOTHING_RETURNED)
        return result

    def executed(self, sql: str, parameters: tuple[Any, ...]) -> int:
        """Rows touched, so a caller can tell an update that found nothing
        from one that worked."""
        with self.transaction() as connection:
            return connection.execute(sql, parameters).rowcount

    def is_reachable(self) -> bool:
        """Whether a `SELECT 1` answers. A connection the server dropped since
        its last use is replaced once: a read is safe to repeat."""
        try:
            return self._is_answering()
        except AppException:
            self._forget_connection()
        try:
            return self._is_answering()
        except AppException:
            return False

    def migrated(self) -> tuple[int, ...]:
        """Bring the schema up to date, and say which versions this call
        applied. Each migration commits with its ledger row, both or neither,
        under an advisory lock so two starting colors never both apply it,
        and under a lock timeout so a busy table fails the start instead of
        queueing every query behind it."""
        result: list[int] = []
        for migration in self.migrations():
            with self.transaction() as connection:
                connection.execute("SELECT pg_advisory_xact_lock(%s)", (Constants.migration_lock_key,))
                connection.execute("SELECT set_config('lock_timeout', %s, true)", (Constants.migration_lock_timeout,))
                connection.execute(self._LEDGER)
                applied = connection.execute(
                    "SELECT version FROM schema_migrations WHERE version = %s", (migration.version,)
                ).fetchone()
                if applied is not None:
                    continue
                for statement in migration.statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations (version, description) VALUES (%s, %s)",
                    (migration.version, migration.description),
                )
                result.append(migration.version)
        return tuple(result)

    def closed(self) -> None:
        with self._opened_lock:
            for connection in self._opened:
                connection.close()
            self._opened = []

    def _is_answering(self) -> bool:
        found = self.row("SELECT 1 AS one", ())
        return found is not None and found["one"] == 1

    def _connection(self) -> psycopg.Connection[DictRow]:
        """This thread's connection, opened on first use and again after the
        server dropped it. Autocommit, so a transaction is always an explicit
        block that commits or rolls back before the connection is reused."""
        current: psycopg.Connection[DictRow] | None = getattr(self._local, "connection", None)
        if current is not None and not current.closed:
            return current
        result = psycopg.connect(
            self._url,
            autocommit=True,
            row_factory=dict_row,
            connect_timeout=Constants.database_connect_timeout_seconds,
            options=f"-c statement_timeout={Constants.database_statement_timeout_ms}",
        )
        self._local.connection = result
        with self._opened_lock:
            self._opened.append(result)
        return result

    def _forget_connection(self) -> None:
        current: psycopg.Connection[DictRow] | None = getattr(self._local, "connection", None)
        if current is None:
            return
        current.close()
        with self._opened_lock:
            self._opened = [connection for connection in self._opened if connection is not current]
        self._local.connection = None
