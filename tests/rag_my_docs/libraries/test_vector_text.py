from __future__ import annotations

import pytest

from rag_my_docs.libraries.vector_text import VectorText


@pytest.mark.parametrize(
    "values, expected",
    [
        pytest.param([0.6, -0.8], "[0.6,-0.8]", id="short"),
        pytest.param([1.0, 0.0], "[1,0]", id="integral"),
        pytest.param([0.123456789012, 1e-12], "[0.12345679,1e-12]", id="eight-significant-digits"),
        pytest.param([], "[]", id="empty"),
    ],
)
def test_of(values: list[float], expected: str) -> None:
    tested = VectorText
    result = tested.of(values)
    assert result == expected
