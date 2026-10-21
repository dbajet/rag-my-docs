from __future__ import annotations

from typing import Any, NamedTuple


class RawBlock(NamedTuple):
    """A text block as PyMuPDF reads it: its lines, where it sits on the page,
    and the font signals a heading is recognised by."""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    lines: tuple[str, ...]
    # The size of most of its characters.
    font_size: float
    # The share of its characters set in bold, 0 to 1.
    bold_share: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "lines": list(self.lines),
            "font_size": self.font_size,
            "bold_share": self.bold_share,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawBlock:
        return cls(
            page=int(data["page"]),
            x0=float(data["x0"]),
            y0=float(data["y0"]),
            x1=float(data["x1"]),
            y1=float(data["y1"]),
            lines=tuple(str(line) for line in data["lines"]),
            font_size=float(data["font_size"]),
            bold_share=float(data["bold_share"]),
        )
