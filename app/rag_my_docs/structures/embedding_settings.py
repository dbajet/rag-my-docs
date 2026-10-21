from __future__ import annotations

from typing import Any, NamedTuple


class EmbeddingSettings(NamedTuple):
    """The local embedding model, and how texts are given to it."""

    model: str
    # Prepended to a question, never to a passage: the model's own convention.
    query_prefix: str
    batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "query_prefix": self.query_prefix, "batch_size": self.batch_size}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingSettings:
        return cls(model=str(data["model"]), query_prefix=str(data["query_prefix"]), batch_size=int(data["batch_size"]))
