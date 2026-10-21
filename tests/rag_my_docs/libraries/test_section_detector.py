from __future__ import annotations

import pytest

from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.libraries.section_detector import SectionDetector
from rag_my_docs.structures.heading import Heading
from rag_my_docs.structures.numbering import Numbering
from rag_my_docs.structures.outline_entry import OutlineEntry
from rag_my_docs.structures.section import Section
from rag_my_docs.structures.text_block import TextBlock

BODY = "The body text of the document goes on for a while, as body text does."


def helper_block(block_id: int, page: int, text: str, size: float, bold: float) -> TextBlock:
    return TextBlock(
        block_id=block_id,
        page=page,
        kind=BlockKind.PARAGRAPH,
        text=text,
        font_size=size,
        bold_share=bold,
        line_count=1,
        top=0.3,
        bottom=0.35,
    )


def helper_numbered_blocks() -> list[TextBlock]:
    return [
        helper_block(0, 1, "1. Introduction", 10.0, 1.0),
        helper_block(1, 1, BODY, 10.0, 0.0),
        helper_block(2, 1, "1.1 Background", 10.0, 1.0),
        helper_block(3, 2, BODY, 10.0, 0.0),
        helper_block(4, 3, "2. Results", 10.0, 1.0),
        helper_block(5, 3, BODY, 10.0, 0.0),
    ]


def helper_section(
    position: int,
    parent: int | None,
    level: int,
    title: str,
    heading: int | None,
    blocks: tuple[int, ...],
    pages: tuple[int, int],
) -> Section:
    return Section(
        position=position,
        parent_position=parent,
        level=level,
        number="",
        title=title,
        source=SectionSource.LAYOUT,
        heading_block_id=heading,
        block_ids=blocks,
        start_page=pages[0],
        end_page=pages[1],
    )


def helper_heading(block_id: int | None, start: int, level: int, title: str) -> Heading:
    return Heading(
        block_id=block_id, start_block_id=start, level=level, number="", title=title, source=SectionSource.LAYOUT
    )


def helper_summary(
    sections: list[Section],
) -> list[tuple[int, int | None, int, str, str, str, tuple[int, ...], int, int]]:
    return [
        (
            section.position,
            section.parent_position,
            section.level,
            section.number,
            section.title,
            section.source.value,
            section.block_ids,
            section.start_page,
            section.end_page,
        )
        for section in sections
    ]


def test_detected() -> None:
    tested = SectionDetector()
    blocks, sections = tested.detected(helper_numbered_blocks(), ())
    result = ([block.kind.value for block in blocks], helper_summary(sections))
    expected = (
        ["heading", "paragraph", "heading", "paragraph", "heading", "paragraph"],
        [
            (0, None, 1, "1", "1. Introduction", "numbered", (1,), 1, 2),
            (1, 0, 2, "1.1", "1.1 Background", "numbered", (3,), 1, 2),
            (2, None, 1, "2", "2. Results", "numbered", (5,), 3, 3),
        ],
    )
    assert result == expected


def test_detected__bookmarks_first() -> None:
    blocks = [
        helper_block(0, 1, "Overview", 10.0, 0.0),
        helper_block(1, 1, BODY, 10.0, 0.0),
        helper_block(2, 2, "1. Details", 10.0, 1.0),
        helper_block(3, 2, BODY, 10.0, 0.0),
        helper_block(4, 3, "2. More", 10.0, 1.0),
    ]
    outline = (OutlineEntry(level=1, title="Overview", page=1), OutlineEntry(level=2, title="Details", page=2))
    tested = SectionDetector()
    _, sections = tested.detected(blocks, outline)
    result = helper_summary(sections)
    expected = [
        (0, None, 1, "", "Overview", "bookmark", (1,), 1, 3),
        (1, 0, 2, "1", "1. Details", "bookmark", (3, 4), 2, 3),
    ]
    assert result == expected


def test_detected__layout_then_fallback() -> None:
    layout = [
        helper_block(0, 1, "Big Title", 18.0, 0.0),
        helper_block(1, 1, BODY, 10.0, 0.0),
        helper_block(2, 1, "Smaller Part", 14.0, 0.0),
        helper_block(3, 2, BODY, 10.0, 0.0),
    ]
    plain = [helper_block(0, 1, BODY, 10.0, 0.0), helper_block(1, 2, BODY, 10.0, 0.0)]
    tested = SectionDetector()
    result = (helper_summary(tested.detected(layout, ())[1]), helper_summary(tested.detected(plain, ())[1]))
    expected = (
        [
            (0, None, 1, "", "Big Title", "layout", (1,), 1, 2),
            (1, 0, 2, "", "Smaller Part", "layout", (3,), 1, 2),
        ],
        [(0, None, 1, "", "", "fallback", (0, 1), 1, 2)],
    )
    assert result == expected


@pytest.mark.parametrize(
    "blocks, expected",
    [
        pytest.param(
            [helper_block(0, 1, "x" * 30, 10.0, 0.0), helper_block(1, 1, "y" * 10, 12.0, 0.0)], 10.0, id="most-chars"
        ),
        pytest.param([], 0.0, id="none"),
        pytest.param(
            [
                helper_block(0, 1, "x" * 30, 10.0, 0.0)._replace(kind=BlockKind.TABLE),
                helper_block(1, 1, "y", 12.0, 0.0),
            ],
            12.0,
            id="paragraphs-only",
        ),
    ],
)
def test_body_size(blocks: list[TextBlock], expected: float) -> None:
    tested = SectionDetector
    result = tested.body_size(blocks)
    assert result == expected


@pytest.mark.parametrize(
    "block, expected",
    [
        pytest.param(helper_block(0, 1, "Scope of Work", 10.0, 1.0), True, id="short-title"),
        pytest.param(helper_block(0, 1, "This is a sentence.", 10.0, 1.0), False, id="ends-like-a-sentence"),
        pytest.param(helper_block(0, 1, "The following applies:", 10.0, 1.0), False, id="colon"),
        pytest.param(helper_block(0, 1, "lowercase start", 10.0, 1.0), False, id="lowercase"),
        pytest.param(helper_block(0, 1, "2024", 10.0, 1.0), False, id="no-letter"),
        pytest.param(helper_block(0, 1, "T" * 121, 10.0, 1.0), False, id="too-long"),
        pytest.param(helper_block(0, 1, "Title", 10.0, 1.0)._replace(line_count=3), False, id="too-many-lines"),
        pytest.param(helper_block(0, 1, "Title", 10.0, 1.0)._replace(kind=BlockKind.HEADER), False, id="header"),
    ],
)
def test_is_candidate(block: TextBlock, expected: bool) -> None:
    tested = SectionDetector
    result = tested.is_candidate(block)
    assert result is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param(
            "2.1 Scope", Numbering(number="2.1", title="Scope", rank=2, depth=2, is_keyword=False), id="dotted"
        ),
        pytest.param(
            "3. Results", Numbering(number="3", title="Results", rank=2, depth=1, is_keyword=False), id="trailing-dot"
        ),
        pytest.param(
            "Section 3", Numbering(number="Section 3", title="", rank=2, depth=1, is_keyword=True), id="section"
        ),
        pytest.param(
            "Article IV: Duties",
            Numbering(number="Article IV", title="Duties", rank=2, depth=1, is_keyword=True),
            id="article-roman",
        ),
        pytest.param(
            "PART I Overview", Numbering(number="PART I", title="Overview", rank=0, depth=1, is_keyword=True), id="part"
        ),
        pytest.param(
            "Chapter 2.3 Methods",
            Numbering(number="Chapter 2.3", title="Methods", rank=1, depth=2, is_keyword=True),
            id="chapter-dotted",
        ),
        pytest.param(
            "Appendix B Tables",
            Numbering(number="Appendix B", title="Tables", rank=2, depth=1, is_keyword=True),
            id="appendix",
        ),
        pytest.param("1.5 million people", None, id="lowercase-title"),
        pytest.param("2024 Results", None, id="year-is-not-a-number"),
        pytest.param("Introduction", None, id="no-number"),
    ],
)
def test_numbering(text: str, expected: Numbering | None) -> None:
    tested = SectionDetector
    result = tested.numbering(text)
    assert result == expected


def test_from_bookmarks() -> None:
    blocks = [
        helper_block(0, 1, "1 Introduction", 10.0, 0.0),
        helper_block(1, 1, BODY, 10.0, 0.0),
        helper_block(2, 2, BODY, 10.0, 0.0)._replace(kind=BlockKind.HEADER),
        helper_block(3, 2, BODY, 10.0, 0.0),
        helper_block(4, 3, "Results", 10.0, 0.0),
        helper_block(5, 3, "Summary", 10.0, 0.0),
    ]
    # Three of five found: the 60% threshold, exactly.
    outline = (
        OutlineEntry(level=1, title="Introduction", page=1),
        OutlineEntry(level=1, title="Not in the text", page=2),
        OutlineEntry(level=1, title="Results", page=3),
        OutlineEntry(level=2, title="Summary", page=3),
        OutlineEntry(level=1, title="Beyond the end", page=9),
    )
    tested = SectionDetector()
    result = tested.from_bookmarks(blocks, outline)
    expected = [
        Heading(
            block_id=0, start_block_id=0, level=1, number="1", title="1 Introduction", source=SectionSource.BOOKMARK
        ),
        Heading(
            block_id=None, start_block_id=3, level=1, number="", title="Not in the text", source=SectionSource.BOOKMARK
        ),
        Heading(block_id=4, start_block_id=4, level=1, number="", title="Results", source=SectionSource.BOOKMARK),
        Heading(block_id=5, start_block_id=5, level=2, number="", title="Summary", source=SectionSource.BOOKMARK),
    ]
    assert result == expected


@pytest.mark.parametrize(
    "outline",
    [
        pytest.param((), id="no-bookmarks"),
        pytest.param(
            (
                OutlineEntry(level=1, title="Introduction", page=1),
                OutlineEntry(level=1, title="Missing one", page=1),
                OutlineEntry(level=1, title="Missing two", page=1),
            ),
            id="under-60-percent-found",
        ),
    ],
)
def test_from_bookmarks__unreliable(outline: tuple[OutlineEntry, ...]) -> None:
    blocks = [helper_block(0, 1, "Introduction", 10.0, 0.0), helper_block(1, 1, BODY, 10.0, 0.0)]
    tested = SectionDetector()
    result = tested.from_bookmarks(blocks, outline)
    expected: list[Heading] = []
    assert result == expected


@pytest.mark.parametrize(
    "entry, cursor, expected",
    [
        pytest.param(OutlineEntry(level=1, title="Scope", page=1), 0, 1, id="same-page"),
        pytest.param(OutlineEntry(level=1, title="Later", page=1), 0, 3, id="next-page"),
        pytest.param(OutlineEntry(level=1, title="Scope", page=1), 2, None, id="before-cursor"),
        pytest.param(OutlineEntry(level=1, title="Far", page=1), 0, None, id="too-far"),
        pytest.param(OutlineEntry(level=1, title="Scope", page=3), 0, None, id="earlier-page-skipped"),
    ],
)
def test_bookmark_block(entry: OutlineEntry, cursor: int, expected: int | None) -> None:
    blocks = [
        helper_block(0, 1, BODY, 10.0, 0.0),
        helper_block(1, 1, "2. SCOPE", 10.0, 1.0),
        helper_block(2, 2, BODY, 10.0, 0.0),
        helper_block(3, 2, "Later", 10.0, 1.0),
        helper_block(4, 4, "Far", 10.0, 1.0),
    ]
    tested = SectionDetector()
    found = tested.bookmark_block(blocks, entry, cursor)
    result = None if found is None else found.block_id
    assert result == expected


@pytest.mark.parametrize(
    "page, cursor, expected",
    [
        pytest.param(2, 0, 2, id="skips-header"),
        pytest.param(1, 0, 0, id="first"),
        pytest.param(3, 0, None, id="empty-page"),
        pytest.param(9, 0, None, id="past-the-end"),
    ],
)
def test_first_block_on(page: int, cursor: int, expected: int | None) -> None:
    blocks = [
        helper_block(0, 1, BODY, 10.0, 0.0),
        helper_block(1, 2, "Header", 8.0, 0.0)._replace(kind=BlockKind.HEADER),
        helper_block(2, 2, BODY, 10.0, 0.0),
        helper_block(3, 4, BODY, 10.0, 0.0),
    ]
    tested = SectionDetector
    found = tested.first_block_on(blocks, page, cursor)
    result = None if found is None else found.block_id
    assert result == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("2.1  Scope & Aims!", "scope aims", id="numbered"),
        pytest.param("Section 3", "section 3", id="keyword-without-title"),
        pytest.param("Overview", "overview", id="plain"),
    ],
)
def test_bare(text: str, expected: str) -> None:
    tested = SectionDetector
    result = tested.bare(text)
    assert result == expected


def test_numbered() -> None:
    blocks = [
        helper_block(0, 1, "PART I Overview", 12.0, 1.0),
        helper_block(1, 1, "Chapter 2 Methods", 10.0, 0.0),
        helper_block(2, 1, "1. Buy milk", 10.0, 0.0),
        helper_block(3, 1, "1.1 Larger font", 10.6, 0.0),
        helper_block(4, 1, "Section 3", 10.0, 0.0)._replace(line_count=2),
        helper_block(5, 1, BODY, 10.0, 0.0),
    ]
    tested = SectionDetector()
    result = tested.numbered(blocks, 10.0)
    expected = [
        Heading(
            block_id=0,
            start_block_id=0,
            level=1,
            number="PART I",
            title="PART I Overview",
            source=SectionSource.NUMBERED,
        ),
        Heading(
            block_id=1,
            start_block_id=1,
            level=11,
            number="Chapter 2",
            title="Chapter 2 Methods",
            source=SectionSource.NUMBERED,
        ),
        Heading(
            block_id=3, start_block_id=3, level=22, number="1.1", title="1.1 Larger font", source=SectionSource.NUMBERED
        ),
    ]
    assert result == expected


def test_numbered__needs_two_headings() -> None:
    blocks = [helper_block(0, 1, "1. Only one", 10.0, 1.0), helper_block(1, 1, BODY, 10.0, 0.0)]
    tested = SectionDetector()
    result = tested.numbered(blocks, 10.0)
    expected: list[Heading] = []
    assert result == expected


def test_by_layout() -> None:
    blocks = [
        helper_block(0, 1, "Title", 20.0, 0.0),
        helper_block(1, 1, "Part", 14.2, 0.0),
        helper_block(2, 1, "Bold Label", 10.0, 1.0),
        helper_block(3, 1, BODY, 10.0, 0.0),
        helper_block(4, 1, "Other Part", 13.9, 0.0),
        helper_block(5, 1, "A paragraph with bold words in it.", 10.0, 1.0),
        helper_block(6, 1, BODY, 10.0, 0.0),
    ]
    tested = SectionDetector()
    result = [(heading.block_id, heading.level) for heading in tested.by_layout(blocks, 10.0)]
    expected = [(0, 1), (1, 2), (2, 3), (4, 2)]
    assert result == expected


def test_by_layout__needs_two_headings() -> None:
    blocks = [helper_block(0, 1, "Title", 20.0, 0.0), helper_block(1, 1, BODY, 10.0, 0.0)]
    tested = SectionDetector()
    result = tested.by_layout(blocks, 10.0)
    expected: list[Heading] = []
    assert result == expected


@pytest.mark.parametrize(
    "blocks, expected",
    [
        pytest.param(
            [helper_block(0, 1, "Label", 10.0, 1.0), helper_block(1, 1, BODY, 10.0, 0.0)], True, id="label-then-text"
        ),
        pytest.param(
            [
                helper_block(0, 1, "Label", 10.0, 1.0),
                helper_block(1, 1, "Footer", 8.0, 0.0)._replace(kind=BlockKind.FOOTER),
                helper_block(2, 2, BODY, 10.0, 0.0),
            ],
            True,
            id="furniture-skipped",
        ),
        pytest.param([helper_block(0, 1, "Label", 10.0, 1.0)], False, id="nothing-after"),
        pytest.param(
            [helper_block(0, 1, "Label", 10.0, 1.0), helper_block(1, 1, "Next Bold", 10.0, 1.0)], False, id="bold-after"
        ),
        pytest.param(
            [helper_block(0, 1, "Label", 10.0, 0.5), helper_block(1, 1, BODY, 10.0, 0.0)], False, id="partly-bold"
        ),
        pytest.param(
            [helper_block(0, 1, "Label", 8.0, 1.0), helper_block(1, 1, BODY, 10.0, 0.0)], False, id="smaller-than-body"
        ),
        pytest.param(
            [helper_block(0, 1, "Label", 10.0, 1.0)._replace(line_count=2), helper_block(1, 1, BODY, 10.0, 0.0)],
            False,
            id="two-lines",
        ),
        pytest.param(
            [
                helper_block(0, 1, "Label", 10.0, 1.0),
                helper_block(1, 1, BODY, 10.0, 0.0)._replace(kind=BlockKind.TABLE),
            ],
            False,
            id="table-after",
        ),
    ],
)
def test_is_bold_label(blocks: list[TextBlock], expected: bool) -> None:
    tested = SectionDetector()
    result = tested.is_bold_label(blocks, 0, 10.0)
    assert result is expected


@pytest.mark.parametrize(
    "size, expected",
    [
        pytest.param(14.2, 14.0, id="down"),
        pytest.param(13.9, 14.0, id="up"),
        pytest.param(10.75, 11.0, id="half-up"),
    ],
)
def test_size_step(size: float, expected: float) -> None:
    tested = SectionDetector
    result = tested.size_step(size)
    assert result == expected


def test_sections_of() -> None:
    blocks = [
        helper_block(0, 1, "Preamble text.", 10.0, 0.0),
        helper_block(1, 1, "Header", 8.0, 0.0)._replace(kind=BlockKind.HEADER),
        helper_block(2, 2, "Part", 14.0, 0.0)._replace(kind=BlockKind.HEADING),
        helper_block(3, 2, "Text a.", 10.0, 0.0),
        helper_block(4, 3, "Sub", 12.0, 0.0)._replace(kind=BlockKind.HEADING),
        helper_block(5, 4, "Text b.", 10.0, 0.0),
        helper_block(6, 5, "Text c.", 10.0, 0.0),
        helper_block(7, 6, "Other", 14.0, 0.0)._replace(kind=BlockKind.HEADING),
        helper_block(8, 6, "Footer", 8.0, 0.0)._replace(kind=BlockKind.FOOTER),
    ]
    headings = [
        helper_heading(2, 2, 30, "Part"),
        helper_heading(4, 4, 50, "Sub"),
        helper_heading(None, 6, 70, "Unmapped bookmark"),
        helper_heading(7, 7, 30, "Other"),
    ]
    tested = SectionDetector()
    result = tested.sections_of(blocks, headings)
    expected = [
        helper_section(0, None, 1, "", None, (0,), (1, 1))._replace(source=SectionSource.FALLBACK),
        helper_section(1, None, 1, "Part", 2, (3,), (2, 5)),
        helper_section(2, 1, 2, "Sub", 4, (5,), (3, 5)),
        helper_section(3, 2, 3, "Unmapped bookmark", None, (6,), (5, 5)),
        helper_section(4, None, 1, "Other", 7, (), (6, 6)),
    ]
    assert result == expected


def test_sections_of__no_block() -> None:
    tested = SectionDetector()
    result = tested.sections_of([], [])
    expected: list[Section] = []
    assert result == expected


def test_opened() -> None:
    tested = SectionDetector
    result = tested.opened(3, 1, 2, helper_heading(7, 7, 20, "Scope")._replace(number="2.1"), 4)
    expected = Section(
        position=3,
        parent_position=1,
        level=2,
        number="2.1",
        title="Scope",
        source=SectionSource.LAYOUT,
        heading_block_id=7,
        block_ids=(),
        start_page=4,
        end_page=4,
    )
    assert result == expected


def test_untitled() -> None:
    tested = SectionDetector
    result = tested.untitled(2)
    expected = Section(
        position=0,
        parent_position=None,
        level=1,
        number="",
        title="",
        source=SectionSource.FALLBACK,
        heading_block_id=None,
        block_ids=(),
        start_page=2,
        end_page=2,
    )
    assert result == expected


def test_spread() -> None:
    sections = [
        helper_section(0, None, 1, "A", 0, (), (3, 3)),
        helper_section(1, 0, 2, "B", 1, (), (4, 5)),
        helper_section(2, 1, 3, "C", 2, (), (2, 9)),
        helper_section(3, None, 1, "D", 3, (), (10, 10)),
    ]
    tested = SectionDetector
    result = [(section.start_page, section.end_page) for section in tested.spread(sections)]
    expected = [(2, 9), (2, 9), (2, 9), (10, 10)]
    assert result == expected
