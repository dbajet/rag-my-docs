from __future__ import annotations

import pytest

from rag_my_docs.structures.answer_citation import AnswerCitation
from rag_my_docs.structures.model_answer import ModelAnswer
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = ModelAnswer
    fields = {"is_found": "bool", "text": "str", "citations": "tuple[AnswerCitation, ...]"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = ModelAnswer(is_found=True, text="$15,750.", citations=(AnswerCitation(2, "Publication 501", 25),))
    result = tested.to_dict()
    expected = {
        "is_found": True,
        "text": "$15,750.",
        "citations": [{"passage": 2, "document": "Publication 501", "page": 25}],
    }
    assert result == expected


def test_from_dict() -> None:
    tested = ModelAnswer
    result = tested.from_dict(
        {"is_found": False, "text": "Not found.", "citations": [{"passage": 1, "document": "a", "page": 3}]}
    )
    expected = ModelAnswer(is_found=False, text="Not found.", citations=(AnswerCitation(1, "a", 3),))
    assert result == expected


def test_from_dict__a_missing_field_raises() -> None:
    tested = ModelAnswer
    with pytest.raises(KeyError):
        tested.from_dict({"is_found": True, "text": "Ada."})
