from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.structures.answer_result import AnswerResult


class EvaluationOutcome(NamedTuple):
    """A test question run end to end, judged in two parts: did retrieval give
    the model a passage from a document it needed (`missing_documents` lists
    them all when none came), and does the answer hold the facts (or, for a
    question the publications cannot answer, say it was not found)."""

    question: str
    missing_documents: tuple[str, ...]
    missing_facts: tuple[str, ...]
    is_retrieval_right: bool
    is_answer_right: bool
    result: AnswerResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "missing_documents": list(self.missing_documents),
            "missing_facts": list(self.missing_facts),
            "is_retrieval_right": self.is_retrieval_right,
            "is_answer_right": self.is_answer_right,
            "result": self.result.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationOutcome:
        return cls(
            question=str(data["question"]),
            missing_documents=tuple(str(document) for document in data["missing_documents"]),
            missing_facts=tuple(str(fact) for fact in data["missing_facts"]),
            is_retrieval_right=bool(data["is_retrieval_right"]),
            is_answer_right=bool(data["is_answer_right"]),
            result=AnswerResult.from_dict(data["result"]),
        )
