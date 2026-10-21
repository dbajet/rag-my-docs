from __future__ import annotations

from rag_my_docs.constants.embedding_constants import EmbeddingConstants
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.secret_sealer import SecretSealer


class SectionPaths:
    """Each section named by its titles from the top of the document down,
    "2. Results › 2.1 Details": search results show it, and every chunk is
    embedded under it, so a table that never names its subject still says
    where it stands."""

    _UNTITLED = "Untitled section"

    def __init__(self, database: Database, sealer: SecretSealer) -> None:
        self._database = database
        self._sealer = sealer

    def paths(self, document_ids: list[int]) -> dict[int, str]:
        """The stored sections' paths, by section id."""
        rows = self._database.rows(
            "SELECT id, parent_id, title_sealed FROM document_sections WHERE document_id = ANY(%s::bigint[])",
            (document_ids,),
        )
        return self.joined(
            {int(row["id"]): self._sealer.opened(bytes(row["title_sealed"])) for row in rows},
            {int(row["id"]): None if row["parent_id"] is None else int(row["parent_id"]) for row in rows},
        )

    @classmethod
    def joined(cls, titles: dict[int, str], parents: dict[int, int | None]) -> dict[int, str]:
        """The path of every section, by the key `titles` and `parents` share."""
        result: dict[int, str] = {}
        for key in titles:
            chain: list[str] = []
            current: int | None = key
            while current is not None:
                chain.insert(0, titles[current] or cls._UNTITLED)
                current = parents[current]
            result[key] = EmbeddingConstants.path_separator.join(chain)
        return result
