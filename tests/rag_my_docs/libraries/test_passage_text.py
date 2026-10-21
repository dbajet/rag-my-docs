from __future__ import annotations

import pytest

from rag_my_docs.libraries.passage_text import PassageText


@pytest.mark.parametrize(
    "document_name, section_path, text, expected",
    [
        pytest.param(
            "prec5560-setup-specifications-en-us.pdf",
            "Specifications of Precision 5560 › Processor",
            "| Processor core count | 6 | 8 |",
            "prec5560-setup-specifications-en-us › Specifications of Precision 5560 › Processor\n\n"
            "| Processor core count | 6 | 8 |",
            id="pdf",
        ),
        pytest.param("notes", "Untitled section", "Text.", "notes › Untitled section\n\nText.", id="no-extension"),
        pytest.param("2017.results.pdf", "Men", "1. Ada", "2017.results › Men\n\n1. Ada", id="dots-in-the-name"),
    ],
)
def test_of(document_name: str, section_path: str, text: str, expected: str) -> None:
    tested = PassageText
    result = tested.of(document_name, section_path, text)
    assert result == expected
