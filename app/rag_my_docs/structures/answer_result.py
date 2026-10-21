from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.structures.model_answer import ModelAnswer
from rag_my_docs.structures.search_hit import SearchHit


class AnswerResult(NamedTuple):
    """A question answered, with every step shown: the passages retrieved
    and their similarity, the exact prompt the model was sent (its
    instructions, then the passages and the question), its answer, and how
    long each step took. Without a passage close enough, no model is
    called: `system`, `prompt` and `model` are empty, and `answer_ms` is 0."""

    question: str
    passages: tuple[SearchHit, ...]
    system: str
    prompt: str
    answer: ModelAnswer
    model: str
    embedding_ms: float
    search_ms: float
    answer_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "passages": [passage.to_dict() for passage in self.passages],
            "system": self.system,
            "prompt": self.prompt,
            "answer": self.answer.to_dict(),
            "model": self.model,
            "embedding_ms": self.embedding_ms,
            "search_ms": self.search_ms,
            "answer_ms": self.answer_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnswerResult:
        return cls(
            question=str(data["question"]),
            passages=tuple(SearchHit.from_dict(passage) for passage in data["passages"]),
            system=str(data["system"]),
            prompt=str(data["prompt"]),
            answer=ModelAnswer.from_dict(data["answer"]),
            model=str(data["model"]),
            embedding_ms=float(data["embedding_ms"]),
            search_ms=float(data["search_ms"]),
            answer_ms=float(data["answer_ms"]),
        )
