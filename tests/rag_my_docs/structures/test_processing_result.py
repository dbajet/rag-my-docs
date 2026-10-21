from __future__ import annotations

import pytest

from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.structures.chunk import Chunk
from rag_my_docs.structures.processing_result import ProcessingResult
from rag_my_docs.structures.section import Section
from rag_my_docs.structures.source_span import SourceSpan
from rag_my_docs.structures.text_block import TextBlock
from tests.conftest import is_namedtuple


def helper_block() -> TextBlock:
    return TextBlock(
        block_id=0,
        page=1,
        kind=BlockKind.PARAGRAPH,
        text="Hello.",
        font_size=10.0,
        bold_share=0.0,
        line_count=1,
        top=0.1,
        bottom=0.12,
    )


def helper_section(heading_block_id: int | None) -> Section:
    return Section(
        position=0,
        parent_position=None,
        level=1,
        number="",
        title="",
        source=SectionSource.FALLBACK,
        heading_block_id=heading_block_id,
        block_ids=(0,),
        start_page=1,
        end_page=1,
    )


def helper_chunk() -> Chunk:
    return Chunk(
        position=0,
        section_position=0,
        kind=ChunkKind.PROSE,
        text="Hello.",
        prefix_chars=0,
        token_count=2,
        start_page=1,
        end_page=1,
        spans=(SourceSpan(block_id=0, start=0, end=6),),
    )


def helper_instance(sections: tuple[Section, ...], chunks: tuple[Chunk, ...]) -> ProcessingResult:
    return ProcessingResult(
        page_count=1,
        blocks=(helper_block(),),
        sections=sections,
        chunks=chunks,
        report={"pages_without_text": []},
    )


def test_class() -> None:
    tested = ProcessingResult
    fields = {
        "page_count": "int",
        "blocks": "tuple[TextBlock, ...]",
        "sections": "tuple[Section, ...]",
        "chunks": "tuple[Chunk, ...]",
        "report": "dict[str, Any]",
    }
    assert is_namedtuple(tested, fields)


@pytest.mark.parametrize(
    "sections, chunks, expected",
    [
        pytest.param((helper_section(None),), (helper_chunk(),), True, id="chunks"),
        pytest.param((helper_section(None), helper_section(4)), (), True, id="headings-only"),
        pytest.param((helper_section(None),), (), False, id="untitled-and-empty"),
        pytest.param((), (), False, id="nothing"),
    ],
)
def test_has_text(sections: tuple[Section, ...], chunks: tuple[Chunk, ...], expected: bool) -> None:
    tested = helper_instance(sections, chunks)
    result = tested.has_text()
    assert result is expected


def test_to_dict() -> None:
    tested = helper_instance((helper_section(None),), (helper_chunk(),))
    result = tested.to_dict()
    expected = {
        "page_count": 1,
        "blocks": [
            {
                "block_id": 0,
                "page": 1,
                "kind": "paragraph",
                "text": "Hello.",
                "font_size": 10.0,
                "bold_share": 0.0,
                "line_count": 1,
                "top": 0.1,
                "bottom": 0.12,
            }
        ],
        "sections": [
            {
                "position": 0,
                "parent_position": None,
                "level": 1,
                "number": "",
                "title": "",
                "source": "fallback",
                "heading_block_id": None,
                "block_ids": [0],
                "start_page": 1,
                "end_page": 1,
            }
        ],
        "chunks": [
            {
                "position": 0,
                "section_position": 0,
                "kind": "prose",
                "text": "Hello.",
                "prefix_chars": 0,
                "token_count": 2,
                "start_page": 1,
                "end_page": 1,
                "spans": [{"block_id": 0, "start": 0, "end": 6}],
            }
        ],
        "report": {"pages_without_text": []},
    }
    assert result == expected


def test_from_dict() -> None:
    tested = ProcessingResult
    source = helper_instance((helper_section(None),), (helper_chunk(),)).to_dict()
    result = tested.from_dict(source | {"page_count": "1"})
    expected = helper_instance((helper_section(None),), (helper_chunk(),))
    assert result == expected
