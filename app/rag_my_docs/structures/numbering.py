from __future__ import annotations

from typing import Any, NamedTuple


class Numbering(NamedTuple):
    """What an explicit heading number says: "2.1 Scope" is number "2.1",
    title "Scope", depth 2. `rank` orders the keywords: a Part holds Chapters,
    a Chapter holds Sections; a bare number ranks with Sections."""

    number: str
    title: str
    rank: int
    depth: int
    is_keyword: bool

    def level(self) -> int:
        """A raw level, comparable between headings of one document only."""
        return self.rank * 10 + self.depth

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "rank": self.rank,
            "depth": self.depth,
            "is_keyword": self.is_keyword,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Numbering:
        return cls(
            number=str(data["number"]),
            title=str(data["title"]),
            rank=int(data["rank"]),
            depth=int(data["depth"]),
            is_keyword=bool(data["is_keyword"]),
        )
