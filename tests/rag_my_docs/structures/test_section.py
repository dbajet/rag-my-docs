from __future__ import annotations

from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.structures.section import Section
from tests.conftest import is_namedtuple


def helper_child() -> Section:
    return Section(
        position=1,
        parent_position=0,
        level=2,
        number="1.1",
        title="1.1 Background",
        source=SectionSource.NUMBERED,
        heading_block_id=3,
        block_ids=(4, 7),
        start_page=1,
        end_page=2,
    )


def helper_untitled() -> Section:
    return Section(
        position=0,
        parent_position=None,
        level=1,
        number="",
        title="",
        source=SectionSource.FALLBACK,
        heading_block_id=None,
        block_ids=(),
        start_page=1,
        end_page=1,
    )


def test_class() -> None:
    tested = Section
    fields = {
        "position": "int",
        "parent_position": "int | None",
        "level": "int",
        "number": "str",
        "title": "str",
        "source": "SectionSource",
        "heading_block_id": "int | None",
        "block_ids": "tuple[int, ...]",
        "start_page": "int",
        "end_page": "int",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = helper_child()
    result = tested.to_dict()
    expected = {
        "position": 1,
        "parent_position": 0,
        "level": 2,
        "number": "1.1",
        "title": "1.1 Background",
        "source": "numbered",
        "heading_block_id": 3,
        "block_ids": [4, 7],
        "start_page": 1,
        "end_page": 2,
    }
    assert result == expected


def test_from_dict() -> None:
    tested = Section
    result = (
        tested.from_dict(
            {
                "position": "1",
                "parent_position": "0",
                "level": "2",
                "number": "1.1",
                "title": "1.1 Background",
                "source": "numbered",
                "heading_block_id": "3",
                "block_ids": ["4", 7],
                "start_page": "1",
                "end_page": 2,
            }
        ),
        tested.from_dict(
            {
                "position": 0,
                "parent_position": None,
                "level": 1,
                "number": "",
                "title": "",
                "source": "fallback",
                "heading_block_id": None,
                "block_ids": [],
                "start_page": 1,
                "end_page": 1,
            }
        ),
    )
    expected = (helper_child(), helper_untitled())
    assert result == expected
