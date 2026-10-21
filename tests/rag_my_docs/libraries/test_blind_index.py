from __future__ import annotations

import pytest

from rag_my_docs.libraries.blind_index import BlindIndex


def test___init__() -> None:
    """The index key is derived from the data key, never the data key itself."""
    tested = BlindIndex("the-data-key")
    result = tested._key.hex()
    expected = "f7804898beb9fd2a909ed7ea53d654fd90d95b8e3a9ddb9aecb92392af140ed4"
    assert result == expected


@pytest.mark.parametrize(
    "data_key, email, expected",
    [
        pytest.param(
            "the-data-key",
            "ada@example.com",
            "e1c76772ce7839bcc26524a96c8adb02a54c8fbe5a7a6af276fcc6ca3944bd23",
            id="plain",
        ),
        pytest.param(
            "the-data-key",
            "  Ada@Example.COM \n",
            "e1c76772ce7839bcc26524a96c8adb02a54c8fbe5a7a6af276fcc6ca3944bd23",
            id="case-and-spaces",
        ),
        pytest.param(
            "another-key",
            "ada@example.com",
            "c8e4b02794a0fa1c4f00ddd4f6712ee62e616b459121b5369c5d54307a814cd1",
            id="another-key",
        ),
    ],
)
def test_digest(data_key: str, email: str, expected: str) -> None:
    tested = BlindIndex(data_key)
    result = tested.digest(email)
    assert result == expected
