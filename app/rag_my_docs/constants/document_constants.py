from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _DocumentConstants:
    # Served from the owner's session only: no shared cache keeps a copy.
    download_cache_control: str = "private, no-store"
    not_found: str = "This document does not exist."


DocumentConstants = _DocumentConstants()
