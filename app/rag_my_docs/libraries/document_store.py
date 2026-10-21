from __future__ import annotations

from typing import Any

from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.structures.document import Document
from rag_my_docs.structures.document_file import DocumentFile


class DocumentStore:
    """The documents, sealed name and content. Every query that
    reaches one names its owner, so an id changed in a URL finds nothing of
    somebody else's (signed in is not allowed)."""

    _COLUMNS = (
        "id, user_id, name_sealed, media_type, size_bytes, processing_status, index_status, "
        "(SELECT count(*) FROM document_chunks c WHERE c.document_id = documents.id) AS chunk_count, "
        "(SELECT count(*) FROM document_chunks c WHERE c.document_id = documents.id AND c.embedding IS NOT NULL) "
        "AS indexed_count, "
        """to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at"""
    )

    def __init__(self, database: Database, sealer: SecretSealer) -> None:
        self._database = database
        self._sealer = sealer

    def created(
        self,
        user_id: int,
        name: str,
        media_type: MediaType,
        content: bytes,
        status: ProcessingStatus,
    ) -> Document:
        row = self._database.returned(
            f"""INSERT INTO documents (user_id, name_sealed, media_type, size_bytes, content_sealed, processing_status)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING {self._COLUMNS}""",
            (
                user_id,
                self._sealer.sealed(name),
                media_type.value,
                len(content),
                self._sealer.sealed_bytes(content),
                status.value,
            ),
        )
        return self._document(row)

    def of_user(self, user_id: int) -> list[Document]:
        """In the order they were stored, the corpus's own; the content stays
        in the table, sealed, until asked for."""
        rows = self._database.rows(
            f"SELECT {self._COLUMNS} FROM documents WHERE user_id = %s ORDER BY id",
            (user_id,),
        )
        return [self._document(row) for row in rows]

    def file_of(self, user_id: int, document_id: int) -> DocumentFile | None:
        row = self._database.row(
            "SELECT name_sealed, media_type, content_sealed FROM documents WHERE id = %s AND user_id = %s",
            (document_id, user_id),
        )
        if row is None:
            return None
        return DocumentFile(
            name=self._sealer.opened(bytes(row["name_sealed"])),
            media_type=MediaType(row["media_type"]),
            content=self._sealer.opened_bytes(bytes(row["content_sealed"])),
        )

    def deleted(self, user_id: int, document_id: int) -> int:
        """Rows removed: none when the document is not this person's."""
        return self._database.executed("DELETE FROM documents WHERE id = %s AND user_id = %s", (document_id, user_id))

    def _document(self, row: dict[str, Any]) -> Document:
        return Document(
            document_id=int(row["id"]),
            user_id=int(row["user_id"]),
            name=self._sealer.opened(bytes(row["name_sealed"])),
            media_type=MediaType(row["media_type"]),
            size_bytes=int(row["size_bytes"]),
            created_at=str(row["created_at"]),
            processing_status=ProcessingStatus(row["processing_status"]),
            index_status=IndexStatus(row["index_status"]),
            chunk_count=int(row["chunk_count"]),
            indexed_count=int(row["indexed_count"]),
        )
