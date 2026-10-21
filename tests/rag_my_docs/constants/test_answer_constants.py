from __future__ import annotations

import dataclasses

from rag_my_docs.constants.answer_constants import AnswerConstants
from tests.conftest import is_dataclass


def test_class() -> None:
    tested = AnswerConstants
    fields = {"min_similarity": "float", "empty_question": "str", "not_covered": "str"}
    assert is_dataclass(tested, fields)


def test_values() -> None:
    tested = AnswerConstants
    result = dataclasses.asdict(tested)
    expected = {
        "min_similarity": 0.66,
        "empty_question": "Type a question to answer.",
        "not_covered": "The publications do not cover this question: no passage is close enough to it.",
    }
    assert result == expected
