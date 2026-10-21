from __future__ import annotations

import pytest

from rag_my_docs.structures.numbering import Numbering
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = Numbering
    fields = {"number": "str", "title": "str", "rank": "int", "depth": "int", "is_keyword": "bool"}
    assert is_namedtuple(tested, fields)


@pytest.mark.parametrize(
    "tested, expected",
    [
        pytest.param(Numbering(number="Part II", title="Rules", rank=0, depth=1, is_keyword=True), 1, id="part"),
        pytest.param(Numbering(number="Chapter 3", title="", rank=1, depth=1, is_keyword=True), 11, id="chapter"),
        pytest.param(Numbering(number="2.1", title="Scope", rank=2, depth=2, is_keyword=False), 22, id="dotted"),
    ],
)
def test_level(tested: Numbering, expected: int) -> None:
    result = tested.level()
    assert result == expected


def test_to_dict() -> None:
    tested = Numbering(number="2.1", title="Scope", rank=2, depth=2, is_keyword=False)
    result = tested.to_dict()
    expected = {"number": "2.1", "title": "Scope", "rank": 2, "depth": 2, "is_keyword": False}
    assert result == expected


def test_from_dict() -> None:
    tested = Numbering
    result = tested.from_dict({"number": "Section 3", "title": "Terms", "rank": "2", "depth": "1", "is_keyword": 1})
    expected = Numbering(number="Section 3", title="Terms", rank=2, depth=1, is_keyword=True)
    assert result == expected
