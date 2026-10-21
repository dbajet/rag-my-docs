from __future__ import annotations

from enum import Enum


class MediaType(Enum):
    PDF = "application/pdf"
    PNG = "image/png"
    JPEG = "image/jpeg"
    WEBP = "image/webp"
    GIF = "image/gif"
    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    CSV = "text/csv"
