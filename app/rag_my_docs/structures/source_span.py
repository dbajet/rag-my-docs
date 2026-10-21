from __future__ import annotations

from typing import Any, NamedTuple


class SourceSpan(NamedTuple):
    """Characters [start, end) of one extracted block: where a chunk's text came from."""

    block_id: int
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {"block_id": self.block_id, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSpan:
        return cls(block_id=int(data["block_id"]), start=int(data["start"]), end=int(data["end"]))
