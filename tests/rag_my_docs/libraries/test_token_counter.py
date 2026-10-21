from __future__ import annotations

import pytest

from rag_my_docs.libraries.token_counter import TokenCounter
from rag_my_docs.structures.text_range import TextRange


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("Dr. Smith paid $3.50.", 9, id="documented-example"),
        pytest.param("", 0, id="empty"),
        pytest.param("   \n\t ", 0, id="whitespace-only"),
        pytest.param("word", 1, id="one-word"),
        pytest.param("e-mail", 3, id="hyphen-is-a-token"),
        pytest.param("2.5%", 4, id="decimal-and-percent"),
        pytest.param("Zoë naïve", 2, id="unicode-letters"),
        pytest.param("snake_case", 1, id="underscore-is-a-word-character"),
    ],
)
def test_count(text: str, expected: int) -> None:
    tested = TokenCounter
    result = tested.count(text)
    assert result == expected


@pytest.mark.parametrize(
    "text, limit, expected",
    [
        pytest.param(
            "Dr. Smith paid $3.50 and more words here",
            4,
            [TextRange(start=0, end=14), TextRange(start=15, end=20), TextRange(start=21, end=40)],
            id="cuts-on-whitespace-never-inside-3.50",
        ),
        pytest.param("one two", 5, [TextRange(start=0, end=7)], id="fits-in-one-window"),
        pytest.param(
            "a b c d e f",
            2,
            [TextRange(start=0, end=3), TextRange(start=4, end=7), TextRange(start=8, end=11)],
            id="exact-windows",
        ),
        pytest.param(
            "aaa.bbb.ccc",
            2,
            [TextRange(start=0, end=4), TextRange(start=4, end=8), TextRange(start=8, end=11)],
            id="no-whitespace-cut-between-tokens",
        ),
        pytest.param("", 3, [], id="empty"),
    ],
)
def test_windows(text: str, limit: int, expected: list[TextRange]) -> None:
    tested = TokenCounter
    result = tested.windows(text, limit)
    assert result == expected


@pytest.mark.parametrize(
    "first, last, expected",
    [
        pytest.param(0, 4, 4, id="whitespace-before-last"),
        pytest.param(0, 3, 1, id="latest-whitespace-boundary"),
        pytest.param(3, 4, 4, id="none-in-window-keeps-last"),
    ],
)
def test__cut(first: int, last: int, expected: int) -> None:
    # "ab cd.ef gh": tokens ab, cd, ., ef, gh
    tokens = [
        TextRange(start=0, end=2),
        TextRange(start=3, end=5),
        TextRange(start=5, end=6),
        TextRange(start=6, end=8),
        TextRange(start=9, end=11),
    ]
    tested = TokenCounter
    result = tested._cut(tokens, first, last)
    assert result == expected
