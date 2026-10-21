from __future__ import annotations

from enum import Enum


class BlockKind(Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"
