from __future__ import annotations

from typing import Any, NamedTuple


class RawTable(NamedTuple):
    """A table PyMuPDF found from its ruling lines: its header row and its rows
    of cells, and the area it covers, so the prose read there is not kept twice."""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "header": list(self.header),
            "rows": [list(row) for row in self.rows],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawTable:
        return cls(
            page=int(data["page"]),
            x0=float(data["x0"]),
            y0=float(data["y0"]),
            x1=float(data["x1"]),
            y1=float(data["y1"]),
            header=tuple(str(cell) for cell in data["header"]),
            rows=tuple(tuple(str(cell) for cell in row) for row in data["rows"]),
        )
