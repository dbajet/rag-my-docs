from __future__ import annotations

from typing import Any, NamedTuple


class EmbeddingCarry(NamedTuple):
    """An embedding kept across a reprocess for a chunk whose text did not
    change: the vector as pgvector prints it, and the model that made it."""

    vector: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {"vector": self.vector, "model": self.model}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingCarry:
        return cls(vector=str(data["vector"]), model=str(data["model"]))
