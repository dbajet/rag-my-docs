from __future__ import annotations

from enum import Enum


class SectionSource(Enum):
    BOOKMARK = "bookmark"
    NUMBERED = "numbered"
    LAYOUT = "layout"
    FALLBACK = "fallback"
