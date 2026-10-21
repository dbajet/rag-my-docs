from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.constants.processing_constants import ProcessingConstants


class ChunkingSettings(NamedTuple):
    """How sentences are grouped into chunks, in tokens of `TokenCounter`."""

    target_tokens: int
    max_tokens: int
    min_tokens: int
    overlap_sentences: int

    def strategy(self) -> str:
        """Stored with every chunk: a document chunked under other settings is
        recognisably stale and can be processed again."""
        return (
            f"{ProcessingConstants.strategy_name} target={self.target_tokens} max={self.max_tokens} "
            f"min={self.min_tokens} overlap={self.overlap_sentences}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_tokens": self.target_tokens,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "overlap_sentences": self.overlap_sentences,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkingSettings:
        return cls(
            target_tokens=int(data["target_tokens"]),
            max_tokens=int(data["max_tokens"]),
            min_tokens=int(data["min_tokens"]),
            overlap_sentences=int(data["overlap_sentences"]),
        )
