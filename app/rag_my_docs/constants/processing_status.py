from __future__ import annotations

from enum import Enum


class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_TEXT = "no_text"
    UNSUPPORTED = "unsupported"
