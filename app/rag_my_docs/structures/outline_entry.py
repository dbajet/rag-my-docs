from __future__ import annotations

from typing import Any, NamedTuple


class OutlineEntry(NamedTuple):
    """One PDF bookmark: its depth, its title and the page it points to."""

    level: int
    title: str
    page: int

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "title": self.title, "page": self.page}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutlineEntry:
        return cls(level=int(data["level"]), title=str(data["title"]), page=int(data["page"]))
