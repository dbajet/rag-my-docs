from __future__ import annotations

from rag_my_docs.structures.answer_citation import AnswerCitation
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = AnswerCitation
    fields = {"passage": "int", "document": "str", "page": "int"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = AnswerCitation(passage=3, document="Publication 17", page=2)
    result = tested.to_dict()
    expected = {"passage": 3, "document": "Publication 17", "page": 2}
    assert result == expected


def test_from_dict() -> None:
    tested = AnswerCitation
    result = tested.from_dict({"passage": "3", "document": "Publication 17", "page": "2"})
    expected = AnswerCitation(passage=3, document="Publication 17", page=2)
    assert result == expected
