from __future__ import annotations

import pytest

from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.libraries.page_furniture import PageFurniture
from rag_my_docs.structures.text_block import TextBlock


def helper_block(block_id: int, page: int, text: str, top: float, kind: BlockKind) -> TextBlock:
    return TextBlock(
        block_id=block_id,
        page=page,
        kind=kind,
        text=text,
        font_size=8.0,
        bold_share=0.0,
        line_count=1,
        top=top,
        bottom=top + 0.02,
    )


def helper_kinds(blocks: list[TextBlock]) -> list[tuple[int, str]]:
    return [(block.block_id, block.kind.value) for block in blocks]


def test_marked() -> None:
    paragraph = BlockKind.PARAGRAPH
    blocks = [
        helper_block(0, 1, "ACME Annual Report", 0.03, paragraph),
        helper_block(1, 1, "Introduction", 0.05, paragraph),
        helper_block(2, 1, "Body text of page one.", 0.4, paragraph),
        helper_block(3, 1, "Page 1 of 3", 0.95, paragraph),
        helper_block(4, 2, "ACME  annual report", 0.03, paragraph),
        helper_block(5, 2, "Body text of page two.", 0.4, paragraph),
        helper_block(6, 2, "Page 2 of 3", 0.95, paragraph),
        helper_block(7, 3, "ACME Annual Report", 0.03, paragraph),
        helper_block(8, 3, "Page 3 of 3", 0.95, paragraph),
        helper_block(9, 3, "| a | b |", 0.96, BlockKind.TABLE),
    ]
    tested = PageFurniture()
    result = helper_kinds(tested.marked(blocks))
    expected = [
        (0, "header"),
        (1, "paragraph"),
        (2, "paragraph"),
        (3, "footer"),
        (4, "header"),
        (5, "paragraph"),
        (6, "footer"),
        (7, "header"),
        (8, "footer"),
        (9, "table"),
    ]
    assert result == expected


def test_marked__one_page_has_no_furniture() -> None:
    blocks = [
        helper_block(0, 1, "Report", 0.03, BlockKind.PARAGRAPH),
        helper_block(1, 1, "Report", 0.95, BlockKind.PARAGRAPH),
    ]
    tested = PageFurniture()
    result = helper_kinds(tested.marked(blocks))
    expected = [(0, "paragraph"), (1, "paragraph")]
    assert result == expected


def test_marked__needs_half_of_the_pages() -> None:
    paragraph = BlockKind.PARAGRAPH
    blocks = [helper_block(page, page, f"Text {page}.", 0.5, paragraph) for page in range(1, 6)]
    blocks += [helper_block(10, 1, "Draft", 0.03, paragraph), helper_block(11, 2, "Draft", 0.03, paragraph)]
    blocks += [helper_block(12, 1, "Memo", 0.95, paragraph), helper_block(13, 2, "Memo", 0.95, paragraph)]
    blocks += [helper_block(14, 3, "Memo", 0.95, paragraph)]
    tested = PageFurniture()
    result = helper_kinds(tested.marked(blocks))
    expected = [
        (1, "paragraph"),
        (2, "paragraph"),
        (3, "paragraph"),
        (4, "paragraph"),
        (5, "paragraph"),
        (10, "paragraph"),
        (11, "paragraph"),
        (12, "footer"),
        (13, "footer"),
        (14, "footer"),
    ]
    assert result == expected


def test_marked__printed_page_numbers() -> None:
    """A printed book: the number on the left of even pages and on the right of
    odd ones, beside each chapter's own title, after an unnumbered cover. No two
    footers repeat, yet every one carries its page number (PDF page minus one)."""
    paragraph = BlockKind.PARAGRAPH
    blocks = [
        helper_block(0, 1, "Precision 5560 Setup", 0.5, paragraph),
        helper_block(1, 2, "1 Set up your computer", 0.95, paragraph),
        helper_block(2, 3, "Views of Precision 5560 2", 0.95, paragraph),
        helper_block(3, 4, "3 Specifications of Precision 5560", 0.95, paragraph),
        helper_block(4, 5, "Specifications of Precision 5560 4", 0.95, paragraph),
        helper_block(5, 5, "Table 12. Storage specifications", 0.03, paragraph),
        helper_block(6, 6, "Getting help 5", 0.95, paragraph),
        # A number at the edge that is not this page's: kept.
        helper_block(7, 6, "Table 18. Battery (continued) 9", 0.03, paragraph),
    ]
    tested = PageFurniture()
    result = helper_kinds(tested.marked(blocks))
    expected = [
        (0, "paragraph"),
        (1, "footer"),
        (2, "footer"),
        (3, "footer"),
        (4, "footer"),
        (5, "paragraph"),
        (6, "footer"),
        (7, "paragraph"),
    ]
    assert result == expected


@pytest.mark.parametrize(
    "blocks, needed, expected",
    [
        pytest.param(
            [
                helper_block(1, 1, "1 Intro", 0.95, BlockKind.PARAGRAPH),
                helper_block(2, 2, "Intro 2", 0.95, BlockKind.PARAGRAPH),
                helper_block(3, 3, "3", 0.03, BlockKind.PARAGRAPH),
            ],
            2,
            {1, 2, 3},
            id="page-equals-number-at-either-end",
        ),
        pytest.param(
            [
                helper_block(1, 3, "Chapter 1", 0.95, BlockKind.PARAGRAPH),
                helper_block(2, 4, "2 Chapter", 0.95, BlockKind.PARAGRAPH),
                helper_block(3, 5, "Page 5", 0.95, BlockKind.PARAGRAPH),
            ],
            2,
            {1, 2},
            id="one-offset-wins-on-enough-pages",
        ),
        pytest.param(
            [
                helper_block(1, 1, "1 Intro", 0.95, BlockKind.PARAGRAPH),
                helper_block(2, 2, "Intro 2", 0.95, BlockKind.PARAGRAPH),
            ],
            3,
            set(),
            id="too-few-pages",
        ),
        pytest.param(
            [
                helper_block(1, 1, "1 Intro", 0.5, BlockKind.PARAGRAPH),
                helper_block(2, 2, "Intro 2", 0.95, BlockKind.TABLE),
                helper_block(3, 3, "Section 3 of the report", 0.95, BlockKind.PARAGRAPH),
            ],
            1,
            set(),
            id="not-candidates-or-number-inside",
        ),
    ],
)
def test_numbered(blocks: list[TextBlock], needed: int, expected: set[int]) -> None:
    tested = PageFurniture
    result = tested.numbered(blocks, needed)
    assert result == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("14 Specifications of Precision 5560", {14, 5560}, id="both-ends"),
        pytest.param("  Views of Precision 5560 19 ", {19}, id="trailing-trimmed"),
        pytest.param("7", {7}, id="number-alone"),
        pytest.param("Section 3 of the report", set(), id="number-inside"),
        pytest.param("A4 paper 2x", set(), id="digits-in-words"),
    ],
)
def test_edge_numbers(text: str, expected: set[int]) -> None:
    tested = PageFurniture
    result = tested.edge_numbers(text)
    assert result == expected


@pytest.mark.parametrize(
    "block, expected",
    [
        pytest.param(helper_block(0, 1, "x", 0.02, BlockKind.PARAGRAPH), True, id="top-edge"),
        pytest.param(helper_block(0, 1, "x", 0.95, BlockKind.PARAGRAPH), True, id="bottom-edge"),
        pytest.param(helper_block(0, 1, "x", 0.5, BlockKind.PARAGRAPH), False, id="middle"),
        pytest.param(helper_block(0, 1, "x", 0.02, BlockKind.TABLE), False, id="table"),
        pytest.param(helper_block(0, 1, "x", 0.02, BlockKind.PARAGRAPH)._replace(line_count=3), False, id="too-long"),
    ],
)
def test_is_candidate(block: TextBlock, expected: bool) -> None:
    tested = PageFurniture
    result = tested.is_candidate(block)
    assert result is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("Page 12 of 30", "page # of #", id="digits"),
        pytest.param("  ACME   Report ", "acme report", id="spacing-and-case"),
    ],
)
def test_key(text: str, expected: str) -> None:
    tested = PageFurniture
    result = tested.key(text)
    assert result == expected


@pytest.mark.parametrize(
    "top, expected",
    [
        pytest.param(0.02, BlockKind.HEADER, id="header"),
        pytest.param(0.95, BlockKind.FOOTER, id="footer"),
    ],
)
def test_as_furniture(top: float, expected: BlockKind) -> None:
    tested = PageFurniture
    result = tested.as_furniture(helper_block(0, 1, "x", top, BlockKind.PARAGRAPH)).kind
    assert result is expected
