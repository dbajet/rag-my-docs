from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.structures.answer_citation import AnswerCitation


class ModelAnswer(NamedTuple):
    """The model's reply, read: whether the passages held the answer, the
    answer, and the passages it says it used."""

    is_found: bool
    text: str
    citations: tuple[AnswerCitation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_found": self.is_found,
            "text": self.text,
            "citations": [citation.to_dict() for citation in self.citations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelAnswer:
        return cls(
            is_found=bool(data["is_found"]),
            text=str(data["text"]),
            citations=tuple(AnswerCitation.from_dict(citation) for citation in data["citations"]),
        )
