from __future__ import annotations

from typing import Any, NamedTuple


class TextRange(NamedTuple):
    """A half-open range of characters, [start, end), in some text."""

    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextRange:
        return cls(start=int(data["start"]), end=int(data["end"]))
