from __future__ import annotations

from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.structures.heading import Heading
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = Heading
    fields = {
        "block_id": "int | None",
        "start_block_id": "int",
        "level": "int",
        "number": "str",
        "title": "str",
        "source": "SectionSource",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = Heading(block_id=None, start_block_id=7, level=2, number="", title="Annex", source=SectionSource.BOOKMARK)
    result = tested.to_dict()
    expected = {
        "block_id": None,
        "start_block_id": 7,
        "level": 2,
        "number": "",
        "title": "Annex",
        "source": "bookmark",
    }
    assert result == expected


def test_from_dict() -> None:
    tested = Heading
    result = (
        tested.from_dict(
            {
                "block_id": "3",
                "start_block_id": "3",
                "level": "22",
                "number": "2.1",
                "title": "2.1 Scope",
                "source": "numbered",
            }
        ),
        tested.from_dict(
            {
                "block_id": None,
                "start_block_id": 7,
                "level": 1,
                "number": "",
                "title": "Annex",
                "source": "bookmark",
            }
        ),
    )
    expected = (
        Heading(block_id=3, start_block_id=3, level=22, number="2.1", title="2.1 Scope", source=SectionSource.NUMBERED),
        Heading(block_id=None, start_block_id=7, level=1, number="", title="Annex", source=SectionSource.BOOKMARK),
    )
    assert result == expected
