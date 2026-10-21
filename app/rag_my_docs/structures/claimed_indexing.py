from __future__ import annotations

from typing import Any, NamedTuple


class ClaimedIndexing(NamedTuple):
    """A document this process is indexing: `run` is the number of this attempt,
    and only the run that still holds the document may mark it indexed."""

    document_id: int
    run: int
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {"document_id": self.document_id, "run": self.run, "model": self.model}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClaimedIndexing:
        return cls(document_id=int(data["document_id"]), run=int(data["run"]), model=str(data["model"]))
