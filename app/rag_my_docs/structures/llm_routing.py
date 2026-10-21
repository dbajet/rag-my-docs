from __future__ import annotations

from typing import Any, NamedTuple


class LlmRouting(NamedTuple):
    """Which connection answers; None while none is chosen."""

    answers: int | None

    def to_dict(self) -> dict[str, Any]:
        return {"answers": self.answers}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LlmRouting:
        return cls(answers=None if data["answers"] is None else int(data["answers"]))
