from __future__ import annotations

import hashlib
import json

from rag_my_docs.constants.evaluation_constants import EvaluationConstants
from tests.conftest import is_dataclass


def test_class() -> None:
    tested = EvaluationConstants
    fields = {
        "cases": "tuple[tuple[str, tuple[str, ...], tuple[str, ...], bool], ...]",
        "run_out_of_range": "str",
    }
    assert is_dataclass(tested, fields)


def test_values() -> None:
    tested = EvaluationConstants
    # The test set is long: a digest of its exact content, printed on failure.
    text = json.dumps(tested.cases, ensure_ascii=False, indent=1)
    result = (hashlib.md5(text.encode()).hexdigest(), tested.run_out_of_range)
    expected = ("640e3a6f59a625df0752d77587197c1a", "This test question does not exist.")
    assert result == expected, text
    exp_unanswerable = [
        ("What is the California state income tax rate?", (), (), False),
        ("How do I renew a U.S. passport?", (), (), False),
    ]
    assert [case for case in tested.cases if case[3] is False] == exp_unanswerable
