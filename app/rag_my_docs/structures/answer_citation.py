from __future__ import annotations

from typing import Any, NamedTuple


class AnswerCitation(NamedTuple):
    """A passage the model says its answer comes from: its number in the
    prompt, and the publication and page it was taken from."""

    passage: int
    document: str
    page: int

    def to_dict(self) -> dict[str, Any]:
        return {"passage": self.passage, "document": self.document, "page": self.page}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnswerCitation:
        return cls(passage=int(data["passage"]), document=str(data["document"]), page=int(data["page"]))
