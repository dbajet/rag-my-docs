from __future__ import annotations

from rag_my_docs.structures.evaluation_case import EvaluationCase
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = EvaluationCase
    fields = {
        "question": "str",
        "documents": "tuple[str, ...]",
        "facts": "tuple[str, ...]",
        "expects_answer": "bool",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = EvaluationCase("Who won?", ("2016.pdf",), ("Lovelace",), True)
    result = tested.to_dict()
    expected = {"question": "Who won?", "documents": ["2016.pdf"], "facts": ["Lovelace"], "expects_answer": True}
    assert result == expected


def test_from_dict() -> None:
    tested = EvaluationCase
    result = tested.from_dict({"question": "BIOS?", "documents": ["5560.pdf"], "facts": [], "expects_answer": False})
    expected = EvaluationCase("BIOS?", ("5560.pdf",), (), False)
    assert result == expected
