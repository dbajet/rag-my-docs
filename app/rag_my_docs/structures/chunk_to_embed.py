from __future__ import annotations

from typing import Any, NamedTuple


class ChunkToEmbed(NamedTuple):
    """A chunk still without an embedding from the current model, written as
    the passage to embed (`PassageText`): its context, then its opened text."""

    chunk_id: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkToEmbed:
        return cls(chunk_id=int(data["chunk_id"]), text=str(data["text"]))
