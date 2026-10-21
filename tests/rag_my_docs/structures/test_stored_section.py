from __future__ import annotations

from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.structures.stored_section import StoredSection
from tests.conftest import is_namedtuple


def helper_instance(parent_id: int | None, heading_block_id: int | None) -> StoredSection:
    return StoredSection(
        section_id=11,
        parent_id=parent_id,
        position=1,
        level=2,
        number="1.1",
        title="1.1 Background",
        source=SectionSource.NUMBERED,
        start_page=1,
        end_page=2,
        heading_block_id=heading_block_id,
        chunk_count=3,
    )


def test_class() -> None:
    tested = StoredSection
    fields = {
        "section_id": "int",
        "parent_id": "int | None",
        "position": "int",
        "level": "int",
        "number": "str",
        "title": "str",
        "source": "SectionSource",
        "start_page": "int",
        "end_page": "int",
        "heading_block_id": "int | None",
        "chunk_count": "int",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = helper_instance(10, 3)
    result = tested.to_dict()
    expected = {
        "section_id": 11,
        "parent_id": 10,
        "position": 1,
        "level": 2,
        "number": "1.1",
        "title": "1.1 Background",
        "source": "numbered",
        "start_page": 1,
        "end_page": 2,
        "heading_block_id": 3,
        "chunk_count": 3,
    }
    assert result == expected


def test_from_dict() -> None:
    tested = StoredSection
    data = {
        "section_id": "11",
        "parent_id": "10",
        "position": "1",
        "level": "2",
        "number": "1.1",
        "title": "1.1 Background",
        "source": "numbered",
        "start_page": "1",
        "end_page": "2",
        "heading_block_id": "3",
        "chunk_count": "3",
    }
    result = (tested.from_dict(data), tested.from_dict(data | {"parent_id": None, "heading_block_id": None}))
    expected = (helper_instance(10, 3), helper_instance(None, None))
    assert result == expected
