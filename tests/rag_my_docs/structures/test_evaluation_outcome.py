from __future__ import annotations

from rag_my_docs.structures.answer_result import AnswerResult
from rag_my_docs.structures.evaluation_outcome import EvaluationOutcome
from rag_my_docs.structures.model_answer import ModelAnswer
from tests.conftest import is_namedtuple


def helper_result() -> AnswerResult:
    return AnswerResult("Who won?", (), "", "", ModelAnswer(True, "Ada.", ()), "good", 1.0, 2.0, 3.0)


def test_class() -> None:
    tested = EvaluationOutcome
    fields = {
        "question": "str",
        "missing_documents": "tuple[str, ...]",
        "missing_facts": "tuple[str, ...]",
        "is_retrieval_right": "bool",
        "is_answer_right": "bool",
        "result": "AnswerResult",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = EvaluationOutcome("Who won?", ("2016.pdf",), ("Lovelace",), False, False, helper_result())
    result = tested.to_dict()
    expected = {
        "question": "Who won?",
        "missing_documents": ["2016.pdf"],
        "missing_facts": ["Lovelace"],
        "is_retrieval_right": False,
        "is_answer_right": False,
        "result": {
            "question": "Who won?",
            "passages": [],
            "system": "",
            "prompt": "",
            "answer": {"is_found": True, "text": "Ada.", "citations": []},
            "model": "good",
            "embedding_ms": 1.0,
            "search_ms": 2.0,
            "answer_ms": 3.0,
        },
    }
    assert result == expected


def test_from_dict() -> None:
    tested = EvaluationOutcome
    result = tested.from_dict(
        {
            "question": "Who won?",
            "missing_documents": [],
            "missing_facts": [],
            "is_retrieval_right": True,
            "is_answer_right": True,
            "result": helper_result().to_dict(),
        }
    )
    expected = EvaluationOutcome("Who won?", (), (), True, True, helper_result())
    assert result == expected
