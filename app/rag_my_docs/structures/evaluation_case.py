from __future__ import annotations

from typing import Any, NamedTuple


class EvaluationCase(NamedTuple):
    """One question of the test set and what a right answer needs: a passage
    from any of the documents, and the facts the answer must contain, word
    for word (case aside; "a|b" accepts either). A question the publications
    cannot answer expects the answer to say so."""

    question: str
    documents: tuple[str, ...]
    facts: tuple[str, ...]
    expects_answer: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "documents": list(self.documents),
            "facts": list(self.facts),
            "expects_answer": self.expects_answer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationCase:
        return cls(
            question=str(data["question"]),
            documents=tuple(str(document) for document in data["documents"]),
            facts=tuple(str(fact) for fact in data["facts"]),
            expects_answer=bool(data["expects_answer"]),
        )
