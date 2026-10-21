from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.structures.piece import Piece


class PieceGroup(NamedTuple):
    """One chunk in the making: the context it repeats from the chunk before
    (overlap sentences, or a table's header row), then its own pieces."""

    prefix: tuple[Piece, ...]
    pieces: tuple[Piece, ...]

    def token_count(self) -> int:
        return sum(piece.token_count for piece in (*self.prefix, *self.pieces))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix": [piece.to_dict() for piece in self.prefix],
            "pieces": [piece.to_dict() for piece in self.pieces],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PieceGroup:
        return cls(
            prefix=tuple(Piece.from_dict(piece) for piece in data["prefix"]),
            pieces=tuple(Piece.from_dict(piece) for piece in data["pieces"]),
        )
