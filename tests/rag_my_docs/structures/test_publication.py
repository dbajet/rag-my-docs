from __future__ import annotations

from rag_my_docs.structures.publication import Publication
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = Publication
    fields = {"name": "str", "url": "str", "sha256": "str"}
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = Publication(name="Publication 17", url="https://www.irs.gov/pub/irs-pdf/p17.pdf", sha256="ab12")
    result = tested.to_dict()
    expected = {"name": "Publication 17", "url": "https://www.irs.gov/pub/irs-pdf/p17.pdf", "sha256": "ab12"}
    assert result == expected


def test_from_dict() -> None:
    tested = Publication
    result = tested.from_dict(
        {"name": "Publication 17", "url": "https://www.irs.gov/pub/irs-pdf/p17.pdf", "sha256": "ab12"}
    )
    expected = Publication(name="Publication 17", url="https://www.irs.gov/pub/irs-pdf/p17.pdf", sha256="ab12")
    assert result == expected
