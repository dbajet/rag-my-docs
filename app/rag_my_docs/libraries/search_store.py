from __future__ import annotations

from typing import Any

from rag_my_docs.constants.index_status import IndexStatus
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.libraries.vector_text import VectorText
from rag_my_docs.structures.search_hit import SearchHit


class SearchStore:
    """Exact nearest-neighbour search with pgvector: every candidate chunk is
    compared with the question by cosine distance (`<=>`, 0 for the same
    direction, 2 for the opposite) and the closest come first. No approximate
    index: for a few thousand chunks a full scan takes milliseconds and is
    always exact.

    Only the owner's chunks are candidates, and only those embedded by the
    current model: a vector is comparable only with its own model's. A
    deleted document's chunks are gone with it."""

    def __init__(self, database: Database, sealer: SecretSealer, paths: SectionPaths) -> None:
        self._database = database
        self._sealer = sealer
        self._paths = paths

    def indexed_count(self, user_id: int, model: str) -> int:
        row = self._database.returned(
            "SELECT count(*) AS found FROM documents WHERE user_id = %s AND index_status = %s AND index_model = %s",
            (user_id, IndexStatus.INDEXED.value, model),
        )
        return int(row["found"])

    def nearest(
        self,
        user_id: int,
        vector: list[float],
        model: str,
        top_k: int,
    ) -> list[SearchHit]:
        """The `top_k` closest chunks, most similar first; ties keep document order."""
        literal = VectorText.of(vector)
        rows = self._database.rows(
            """SELECT c.id, c.position, c.document_id, d.name_sealed, c.section_id, c.start_page, c.end_page,
                   c.text_sealed, c.prefix_chars, 1 - (c.embedding <=> %s::vector) AS similarity
               FROM document_chunks c JOIN documents d ON d.id = c.document_id
               WHERE d.user_id = %s AND c.embedding IS NOT NULL AND c.embedding_model = %s
               ORDER BY c.embedding <=> %s::vector, c.document_id, c.position
               LIMIT %s""",
            (literal, user_id, model, literal, top_k),
        )
        paths = self._paths.paths(sorted({int(row["document_id"]) for row in rows}))
        return [self.hit(rank, row, paths) for rank, row in enumerate(rows, start=1)]

    def hit(self, rank: int, row: dict[str, Any], paths: dict[int, str]) -> SearchHit:
        return SearchHit(
            rank=rank,
            chunk_id=int(row["id"]),
            chunk_position=int(row["position"]),
            document_id=int(row["document_id"]),
            document_name=self._sealer.opened(bytes(row["name_sealed"])),
            section_id=int(row["section_id"]),
            section_path=paths[int(row["section_id"])],
            start_page=int(row["start_page"]),
            end_page=int(row["end_page"]),
            text=self._sealer.opened(bytes(row["text_sealed"])),
            prefix_chars=int(row["prefix_chars"]),
            similarity=float(row["similarity"]),
        )
