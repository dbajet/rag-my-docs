from __future__ import annotations

from enum import Enum


class IndexStatus(Enum):
    # No chunks to embed: not a PDF, not processed yet, or no text.
    NOT_INDEXED = "not_indexed"
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
