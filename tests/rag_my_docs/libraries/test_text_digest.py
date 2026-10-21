from __future__ import annotations

import pytest

from rag_my_docs.libraries.text_digest import TextDigest

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="


def test___init__() -> None:
    """The digest key is derived from the data key under its own label, never
    the data key itself; the expected values come from `openssl dgst -mac HMAC`."""
    tested = TextDigest(KEY)
    result = tested._key.hex()
    expected = "1fb6942e2e63aac8700bf12023cd6d1054c80aee9b2a9fb5ae355d29bdd210ef"
    assert result == expected


@pytest.mark.parametrize(
    "data_key, text, expected",
    [
        pytest.param(
            KEY,
            "First sentence. Second sentence.",
            "212899c98eaff9cd8eb7c4b230eb5cc13a261fb0d1a63ee5c3a01e9858b224c7",
            id="plain",
        ),
        pytest.param(
            KEY, "Été à Paris", "4552bc1e6a83a95683560546c42fff64f33f5f59d9052c179f4f2cc99d2dc1ce", id="utf-8"
        ),
        pytest.param(
            "another-key",
            "First sentence. Second sentence.",
            "cf216b60d792971af70db909e1ac0ec9f03c6ab4d084fc41acc35d455816a8d1",
            id="another-key",
        ),
    ],
)
def test_digest(data_key: str, text: str, expected: str) -> None:
    tested = TextDigest(data_key)
    result = tested.digest(text)
    assert result == expected
