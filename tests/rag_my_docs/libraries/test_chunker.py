from __future__ import annotations

import pytest

from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.libraries.chunker import Chunker
from rag_my_docs.libraries.sentence_splitter import SentenceSplitter
from rag_my_docs.libraries.token_counter import TokenCounter
from rag_my_docs.structures.chunk import Chunk
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.paragraph import Paragraph
from rag_my_docs.structures.paragraph_segment import ParagraphSegment
from rag_my_docs.structures.piece import Piece
from rag_my_docs.structures.piece_group import PieceGroup
from rag_my_docs.structures.section import Section
from rag_my_docs.structures.source_span import SourceSpan
from rag_my_docs.structures.text_block import TextBlock
from rag_my_docs.structures.text_range import TextRange

# Every sentence of these tests is three words and a period: four tokens.
TABLE = "| Region | Sales |\n| North | 120 |\n| South | 95 |\n| East | 70 |\n| West | 60 |"


def helper_settings(target: int, maximum: int, minimum: int, overlap: int) -> ChunkingSettings:
    return ChunkingSettings(target_tokens=target, max_tokens=maximum, min_tokens=minimum, overlap_sentences=overlap)


def helper_instance(settings: ChunkingSettings) -> Chunker:
    return Chunker(settings, SentenceSplitter(), TokenCounter())


def helper_block(block_id: int, page: int, text: str, kind: BlockKind) -> TextBlock:
    return TextBlock(
        block_id=block_id,
        page=page,
        kind=kind,
        text=text,
        font_size=10.0,
        bold_share=0.0,
        line_count=1,
        top=0.3,
        bottom=0.4,
    )


def helper_section(position: int, block_ids: tuple[int, ...]) -> Section:
    return Section(
        position=position,
        parent_position=None,
        level=1,
        number="",
        title="",
        source=SectionSource.FALLBACK,
        heading_block_id=None,
        block_ids=block_ids,
        start_page=1,
        end_page=1,
    )


def helper_piece(text: str, tokens: int, paragraph: int, lead: str) -> Piece:
    return Piece(
        text=text,
        token_count=tokens,
        spans=(SourceSpan(block_id=paragraph, start=0, end=len(text)),),
        start_page=1,
        end_page=1,
        paragraph=paragraph,
        lead=lead,
    )


def test___init__() -> None:
    settings = helper_settings(200, 350, 75, 1)
    splitter = SentenceSplitter()
    counter = TokenCounter()
    tested = Chunker(settings, splitter, counter)
    result = (tested._settings, tested._splitter, tested._counter)
    expected = (settings, splitter, counter)
    assert result == expected


def test_chunks() -> None:
    blocks = [
        # Three lines: prose, not a caption.
        helper_block(0, 1, "Aa bb cc. Dd ee ff.", BlockKind.PARAGRAPH)._replace(line_count=3),
        helper_block(1, 1, "Table 1. Sales", BlockKind.PARAGRAPH),
        helper_block(2, 1, "| H |\n| r |", BlockKind.TABLE),
        helper_block(3, 2, "Gg hh ii.", BlockKind.PARAGRAPH),
    ]
    tested = helper_instance(helper_settings(200, 350, 75, 1))
    result = tested.chunks(blocks, [helper_section(0, (0, 1, 2)), helper_section(1, (3,))])
    expected = [
        Chunk(
            position=0,
            section_position=0,
            kind=ChunkKind.PROSE,
            text="Aa bb cc. Dd ee ff.",
            prefix_chars=0,
            token_count=8,
            start_page=1,
            end_page=1,
            spans=(SourceSpan(block_id=0, start=0, end=19),),
        ),
        # The caption leaves the prose and leads the table.
        Chunk(
            position=1,
            section_position=0,
            kind=ChunkKind.TABLE,
            text="Table 1. Sales\n\n| H |\n| r |",
            prefix_chars=0,
            token_count=10,
            start_page=1,
            end_page=1,
            spans=(SourceSpan(block_id=1, start=0, end=14), SourceSpan(block_id=2, start=0, end=11)),
        ),
        # Under the minimum, but a chunk never takes text from another section.
        Chunk(
            position=2,
            section_position=1,
            kind=ChunkKind.PROSE,
            text="Gg hh ii.",
            prefix_chars=0,
            token_count=4,
            start_page=2,
            end_page=2,
            spans=(SourceSpan(block_id=3, start=0, end=9),),
        ),
    ]
    assert result == expected


def test_chunks__limits_and_overlap_hold_in_every_section() -> None:
    sentences = [f"Word{number} alpha beta." for number in range(30)]
    blocks = [
        helper_block(0, 1, " ".join(sentences[:12]), BlockKind.PARAGRAPH),
        helper_block(1, 1, " ".join(sentences[12:20]), BlockKind.PARAGRAPH),
        helper_block(2, 2, " ".join(sentences[20:]), BlockKind.PARAGRAPH),
    ]
    settings = helper_settings(20, 30, 8, 1)
    tested = helper_instance(settings)
    chunks = tested.chunks(blocks, [helper_section(0, (0, 1)), helper_section(1, (2,))])
    result = [
        (
            chunk.section_position,
            chunk.token_count <= settings.max_tokens,
            chunk.text[: chunk.prefix_chars].strip(),
            chunk.text[chunk.prefix_chars :].split(" ")[0],
        )
        for chunk in chunks
    ]
    expected = [
        (0, True, "", "Word0"),
        (0, True, "Word4 alpha beta.", "Word5"),
        # Near the target with room to spare: the next paragraph's first sentence joins.
        (0, True, "Word8 alpha beta.", "Word9"),
        (0, True, "Word12 alpha beta.", "Word13"),
        (0, True, "Word16 alpha beta.", "Word17"),
        # A new section starts without overlap; its small last group joined this chunk.
        (1, True, "", "Word20"),
        (1, True, "Word24 alpha beta.", "Word25"),
    ]
    assert result == expected


def test_chunks__long_sentence_is_split_deterministically() -> None:
    text = " ".join(f"w{number}" for number in range(70)) + "."
    tested = helper_instance(helper_settings(20, 30, 8, 1))
    chunks = tested.chunks([helper_block(0, 1, text, BlockKind.PARAGRAPH)], [helper_section(0, (0,))])
    result = [(chunk.text, chunk.prefix_chars, chunk.token_count, chunk.spans) for chunk in chunks]
    expected = [
        (" ".join(f"w{number}" for number in range(30)), 0, 30, (SourceSpan(block_id=0, start=0, end=109),)),
        (" ".join(f"w{number}" for number in range(30, 60)), 0, 30, (SourceSpan(block_id=0, start=110, end=229),)),
        (
            " ".join(f"w{number}" for number in range(60, 70)) + ".",
            0,
            11,
            (SourceSpan(block_id=0, start=230, end=270),),
        ),
    ]
    assert result == expected


def test_chunks__all_source_text_is_accounted_for() -> None:
    """Chunk spans, headings and page furniture together cover every block
    exactly once — whitespace between sentences aside: nothing is lost and
    nothing is doubled except the declared prefixes."""
    long_sentence = " ".join(f"word{number}" for number in range(45)) + "."
    blocks = [
        helper_block(0, 1, "Report header", BlockKind.HEADER),
        helper_block(1, 1, "1. Introduction", BlockKind.HEADING),
        helper_block(2, 1, " ".join(f"First{n} part here." for n in range(9)), BlockKind.PARAGRAPH),
        helper_block(3, 1, "Second paragraph is short. It ends", BlockKind.PARAGRAPH),
        helper_block(4, 1, "Page 1", BlockKind.FOOTER),
        helper_block(5, 2, "Report header", BlockKind.HEADER),
        helper_block(6, 2, "on the next page. " + long_sentence, BlockKind.PARAGRAPH),
        helper_block(7, 2, "2. Figures", BlockKind.HEADING),
        helper_block(8, 2, TABLE, BlockKind.TABLE),
        helper_block(9, 2, "Items:\n- one item.\n- two items.", BlockKind.PARAGRAPH),
        helper_block(10, 2, "Page 2", BlockKind.FOOTER),
    ]
    sections = [helper_section(0, (2, 3, 6)), helper_section(1, (8, 9))]
    tested = helper_instance(helper_settings(20, 30, 8, 1))
    chunks = tested.chunks(blocks, sections)
    covered = {block.block_id: [0] * len(block.text) for block in blocks}
    for chunk in chunks:
        for span in chunk.spans:
            for index in range(span.start, span.end):
                covered[span.block_id][index] += 1
    for block in blocks:
        if block.kind in (BlockKind.HEADING, BlockKind.HEADER, BlockKind.FOOTER):
            covered[block.block_id] = [count + 1 for count in covered[block.block_id]]
    result = {
        block.block_id: (
            sum(
                1
                for index, count in enumerate(covered[block.block_id])
                if count == 0 and not block.text[index].isspace()
            ),
            sum(1 for count in covered[block.block_id] if count > 1),
        )
        for block in blocks
    }
    expected = {block.block_id: (0, 0) for block in blocks}
    assert result == expected
    exp_max = 30
    assert max(chunk.token_count for chunk in chunks if chunk.kind is ChunkKind.PROSE) <= exp_max


@pytest.mark.parametrize(
    "blocks, expected",
    [
        pytest.param(
            [helper_block(0, 1, "Aa bb cc.", BlockKind.PARAGRAPH)],
            [("prose", "Aa bb cc.", 0)],
            id="prose-alone-keeps-its-last-lines",
        ),
        pytest.param(
            [
                helper_block(0, 1, "Table 2. Sales", BlockKind.PARAGRAPH),
                helper_block(1, 1, "| H |\n| r |", BlockKind.TABLE),
                helper_block(2, 1, "Aa bb cc.", BlockKind.PARAGRAPH),
            ],
            [("table", "Table 2. Sales\n\n| H |\n| r |", 0), ("prose", "Aa bb cc.", 0)],
            id="a-caption-alone-leaves-no-prose-chunk",
        ),
        pytest.param(
            [
                helper_block(0, 1, "Aa bb cc.", BlockKind.PARAGRAPH)._replace(line_count=3),
                helper_block(1, 1, "| H |\n| r |", BlockKind.TABLE),
                helper_block(2, 1, "| K |\n| s |", BlockKind.TABLE),
            ],
            [("prose", "Aa bb cc.", 0), ("table", "| H |\n| r |", 0), ("table", "| K |\n| s |", 0)],
            id="no-caption-and-none-carried-to-the-next-table",
        ),
    ],
)
def test_section_chunks(blocks: list[TextBlock], expected: list[tuple[str, str, int]]) -> None:
    tested = helper_instance(helper_settings(200, 350, 75, 1))
    result = [(chunk.kind.value, chunk.text, chunk.prefix_chars) for chunk in tested.section_chunks(0, blocks)]
    assert result == expected


@pytest.mark.parametrize(
    "kinds, expected",
    [
        pytest.param(["p", "p", "t", "p", "t", "t"], [[0, 1], [2], [3], [4], [5]], id="mixed"),
        pytest.param(["t", "p"], [[0], [1]], id="table-first"),
        pytest.param([], [], id="empty"),
    ],
)
def test_runs(kinds: list[str], expected: list[list[int]]) -> None:
    blocks = [
        helper_block(index, 1, "Text.", BlockKind.TABLE if kind == "t" else BlockKind.PARAGRAPH)
        for index, kind in enumerate(kinds)
    ]
    tested = Chunker
    result = [[block.block_id for block in run] for run in tested.runs(blocks)]
    assert result == expected


@pytest.mark.parametrize(
    "run, expected",
    [
        pytest.param(
            [
                helper_block(0, 1, "Long text here.", BlockKind.PARAGRAPH)._replace(line_count=3),
                helper_block(1, 1, "The following table lists sales.", BlockKind.PARAGRAPH),
                helper_block(2, 1, "Table 1. Sales", BlockKind.PARAGRAPH),
            ],
            ([0], [1, 2]),
            id="intro-and-title",
        ),
        pytest.param(
            [helper_block(index, 1, "Short line.", BlockKind.PARAGRAPH) for index in range(3)],
            ([0], [1, 2]),
            id="at-most-two-blocks",
        ),
        pytest.param(
            [
                helper_block(0, 1, "Processor", BlockKind.HEADING),
                helper_block(1, 1, "Table 1. Processor", BlockKind.PARAGRAPH),
            ],
            ([0], [1]),
            id="a-heading-stops-it",
        ),
        pytest.param(
            [helper_block(0, 1, " ".join(["word"] * 61), BlockKind.PARAGRAPH)],
            ([0], []),
            id="over-sixty-tokens",
        ),
        pytest.param(
            [helper_block(0, 1, "Table 1. Sales", BlockKind.PARAGRAPH)],
            ([], [0]),
            id="all-caption",
        ),
        pytest.param(
            [helper_block(0, 1, "Ends with a paragraph.", BlockKind.PARAGRAPH)._replace(line_count=3)],
            ([0], []),
            id="three-lines-is-prose",
        ),
    ],
)
def test_captioned(run: list[TextBlock], expected: tuple[list[int], list[int]]) -> None:
    tested = helper_instance(helper_settings(200, 350, 75, 1))
    prose, caption = tested.captioned(run)
    result = ([block.block_id for block in prose], [block.block_id for block in caption])
    assert result == expected


def test_prose_chunks() -> None:
    blocks = [helper_block(3, 1, "Aa bb cc. Dd ee ff. Gg hh ii. Jj kk ll.", BlockKind.PARAGRAPH)]
    tested = helper_instance(helper_settings(8, 12, 4, 1))
    result = tested.prose_chunks(5, blocks)
    expected = [
        Chunk(
            position=0,
            section_position=5,
            kind=ChunkKind.PROSE,
            text="Aa bb cc. Dd ee ff.",
            prefix_chars=0,
            token_count=8,
            start_page=1,
            end_page=1,
            spans=(SourceSpan(block_id=3, start=0, end=19),),
        ),
        Chunk(
            position=0,
            section_position=5,
            kind=ChunkKind.PROSE,
            text="Dd ee ff. Gg hh ii.",
            prefix_chars=10,
            token_count=8,
            start_page=1,
            end_page=1,
            spans=(SourceSpan(block_id=3, start=20, end=29),),
        ),
        Chunk(
            position=0,
            section_position=5,
            kind=ChunkKind.PROSE,
            text="Gg hh ii. Jj kk ll.",
            prefix_chars=10,
            token_count=8,
            start_page=1,
            end_page=1,
            spans=(SourceSpan(block_id=3, start=30, end=39),),
        ),
    ]
    assert result == expected


def test_paragraphs() -> None:
    blocks = [
        helper_block(0, 1, "First ends here.", BlockKind.PARAGRAPH),
        helper_block(1, 1, "Runs on and", BlockKind.PARAGRAPH),
        helper_block(2, 2, "goes on.", BlockKind.PARAGRAPH),
        helper_block(3, 3, "lower start after an end.", BlockKind.PARAGRAPH),
    ]
    tested = helper_instance(helper_settings(200, 350, 75, 1))
    result = tested.paragraphs(blocks)
    expected = [
        Paragraph(text="First ends here.", segments=(ParagraphSegment(block_id=0, page=1, offset=0, length=16),)),
        Paragraph(
            text="Runs on and goes on.",
            segments=(
                ParagraphSegment(block_id=1, page=1, offset=0, length=11),
                ParagraphSegment(block_id=2, page=2, offset=12, length=8),
            ),
        ),
        Paragraph(
            text="lower start after an end.", segments=(ParagraphSegment(block_id=3, page=3, offset=0, length=25),)
        ),
    ]
    assert result == expected


@pytest.mark.parametrize(
    "previous, block, expected",
    [
        pytest.param((1, "It runs on and"), (2, "goes on."), True, id="page-break-mid-sentence"),
        pytest.param((1, "It runs on and"), (1, "goes on."), False, id="same-page"),
        pytest.param((1, "It runs on and"), (3, "goes on."), False, id="page-gap"),
        pytest.param((1, "It ended."), (2, "goes on."), False, id="sentence-ended"),
        pytest.param((1, "It (ended)"), (2, "goes on."), False, id="bracket-ended"),
        pytest.param((1, "It runs on and"), (2, "Goes on."), False, id="capital-start"),
    ],
)
def test_continues(previous: tuple[int, str], block: tuple[int, str], expected: bool) -> None:
    tested = Chunker
    result = tested.continues(
        helper_block(0, previous[0], previous[1], BlockKind.PARAGRAPH),
        helper_block(1, block[0], block[1], BlockKind.PARAGRAPH),
    )
    assert result is expected


def test_pieces() -> None:
    paragraphs = [
        Paragraph(text="Aa bb. Cc dd.", segments=(ParagraphSegment(block_id=4, page=1, offset=0, length=13),)),
        Paragraph(text="Ee ff gg hh ii jj.", segments=(ParagraphSegment(block_id=5, page=2, offset=0, length=18),)),
    ]
    tested = helper_instance(helper_settings(4, 4, 1, 1))
    result = tested.pieces(paragraphs)
    expected = [
        Piece(
            text="Aa bb.",
            token_count=3,
            spans=(SourceSpan(block_id=4, start=0, end=6),),
            start_page=1,
            end_page=1,
            paragraph=0,
            lead="",
        ),
        Piece(
            text="Cc dd.",
            token_count=3,
            spans=(SourceSpan(block_id=4, start=7, end=13),),
            start_page=1,
            end_page=1,
            paragraph=0,
            lead=" ",
        ),
        Piece(
            text="Ee ff gg hh",
            token_count=4,
            spans=(SourceSpan(block_id=5, start=0, end=11),),
            start_page=2,
            end_page=2,
            paragraph=1,
            lead="",
        ),
        Piece(
            text="ii jj.",
            token_count=3,
            spans=(SourceSpan(block_id=5, start=12, end=18),),
            start_page=2,
            end_page=2,
            paragraph=1,
            lead=" ",
        ),
    ]
    assert result == expected


@pytest.mark.parametrize(
    "start, end, expected",
    [
        pytest.param(4, 13, [TextRange(start=4, end=13)], id="fits"),
        pytest.param(
            14,
            31,
            [TextRange(start=14, end=25), TextRange(start=26, end=31)],
            id="cut-into-windows",
        ),
    ],
)
def test_fitted(start: int, end: int, expected: list[TextRange]) -> None:
    text = "Xx. Aa bb cc. Dd ee ff gg hh ii."
    tested = helper_instance(helper_settings(4, 4, 1, 1))
    result = tested.fitted(text, start, end)
    assert result == expected


def test_piece() -> None:
    paragraph = Paragraph(
        text="Runs on and goes on.",
        segments=(
            ParagraphSegment(block_id=1, page=1, offset=0, length=11),
            ParagraphSegment(block_id=2, page=2, offset=12, length=8),
        ),
    )
    tested = helper_instance(helper_settings(200, 350, 75, 1))
    result = tested.piece(paragraph, 6, 5, 20, " ")
    expected = Piece(
        text="on and goes on.",
        token_count=5,
        spans=(SourceSpan(block_id=1, start=5, end=11), SourceSpan(block_id=2, start=0, end=8)),
        start_page=1,
        end_page=2,
        paragraph=6,
        lead=" ",
    )
    assert result == expected


@pytest.mark.parametrize(
    "pieces, settings, expected",
    [
        pytest.param(
            [helper_piece("a", 4, 0, ""), helper_piece("b", 3, 1, ""), helper_piece("c", 4, 1, " ")],
            helper_settings(6, 12, 4, 0),
            [((), ("a",)), ((), ("b", "c"))],
            id="paragraph-boundary-preferred",
        ),
        pytest.param(
            [helper_piece("a", 2, 0, ""), helper_piece("b", 2, 1, ""), helper_piece("c", 2, 2, "")],
            helper_settings(10, 12, 4, 1),
            [((), ("a", "b", "c"))],
            id="several-paragraphs-in-one-chunk",
        ),
        pytest.param(
            [helper_piece("a", 5, 0, ""), helper_piece("b", 5, 0, " "), helper_piece("c", 5, 0, " ")],
            helper_settings(5, 12, 1, 1),
            [((), ("a",)), (("a",), ("b",)), (("b",), ("c",))],
            id="overlap-one-sentence",
        ),
        pytest.param([], helper_settings(5, 12, 1, 1), [], id="nothing"),
    ],
)
def test_grouped(pieces: list[Piece], settings: ChunkingSettings, expected: list[tuple[tuple[str, ...], ...]]) -> None:
    tested = helper_instance(settings)
    result = [
        (tuple(piece.text for piece in group.prefix), tuple(piece.text for piece in group.pieces))
        for group in tested.grouped(pieces)
    ]
    assert result == expected


@pytest.mark.parametrize(
    "tokens, piece, expected",
    [
        pytest.param(10, helper_piece("x", 3, 0, " "), True, id="would-pass-the-maximum"),
        pytest.param(8, helper_piece("x", 1, 0, " "), True, id="target-reached"),
        pytest.param(5, helper_piece("x", 4, 1, ""), True, id="new-paragraph-near-target"),
        pytest.param(3, helper_piece("x", 6, 1, ""), False, id="new-paragraph-under-minimum"),
        pytest.param(5, helper_piece("x", 2, 1, ""), False, id="new-paragraph-under-target"),
        pytest.param(5, helper_piece("x", 4, 0, " "), False, id="same-paragraph-under-maximum"),
    ],
)
def test_is_full(tokens: int, piece: Piece, expected: bool) -> None:
    group = PieceGroup(prefix=(), pieces=(helper_piece("g", tokens, 0, ""),))
    tested = helper_instance(helper_settings(8, 12, 4, 1))
    result = tested.is_full(group, piece)
    assert result is expected


@pytest.mark.parametrize(
    "overlap, next_tokens, expected",
    [
        pytest.param(1, 4, ("c",), id="one-sentence"),
        pytest.param(2, 4, ("b", "c"), id="two-sentences"),
        pytest.param(0, 4, (), id="no-overlap"),
        pytest.param(1, 9, (), id="dropped-when-over-the-maximum"),
    ],
)
def test_next_group(overlap: int, next_tokens: int, expected: tuple[str, ...]) -> None:
    closed = PieceGroup(
        prefix=(),
        pieces=(helper_piece("a", 4, 0, ""), helper_piece("b", 4, 0, " "), helper_piece("c", 4, 0, " ")),
    )
    tested = helper_instance(helper_settings(8, 12, 4, overlap))
    found = tested.next_group(closed, helper_piece("d", next_tokens, 0, " "))
    result = (tuple(piece.text for piece in found.prefix), found.pieces)
    exp_result = (expected, ())
    assert result == exp_result


@pytest.mark.parametrize(
    "groups, expected",
    [
        pytest.param([((), ((("a", 6),)))], [((), ("a",))], id="single-group"),
        pytest.param(
            [((), (("a", 6),)), (("a",), (("b", 2),))],
            [((), ("a", "b"))],
            id="small-tail-merged",
        ),
        pytest.param(
            [((), (("a", 6),)), (("a",), (("b", 4),))],
            [((), ("a",)), (("a",), ("b",))],
            id="tail-at-minimum-kept",
        ),
        pytest.param(
            [((), (("a", 11),)), (("a",), (("b", 2),))],
            [((), ("a",)), (("a",), ("b",))],
            id="merge-would-pass-the-maximum",
        ),
    ],
)
def test_merged_tail(
    groups: list[tuple[tuple[str, ...], tuple[tuple[str, int], ...]]],
    expected: list[tuple[tuple[str, ...], tuple[str, ...]]],
) -> None:
    sizes = {name: tokens for _, pieces in groups for name, tokens in pieces}
    built = [
        PieceGroup(
            prefix=tuple(helper_piece(name, sizes[name], 0, "") for name in prefix),
            pieces=tuple(helper_piece(name, tokens, 0, " ") for name, tokens in pieces),
        )
        for prefix, pieces in groups
    ]
    tested = helper_instance(helper_settings(8, 12, 4, 1))
    result = [
        (tuple(piece.text for piece in group.prefix), tuple(piece.text for piece in group.pieces))
        for group in tested.merged_tail(built)
    ]
    assert result == expected


def test_table_chunks() -> None:
    tested = helper_instance(helper_settings(10, 12, 1, 1))
    result = tested.table_chunks(2, helper_block(8, 3, TABLE, BlockKind.TABLE), [])
    header = "| Region | Sales |\n"
    expected = [
        Chunk(
            position=0,
            section_position=2,
            kind=ChunkKind.TABLE,
            text="| Region | Sales |\n| North | 120 |",
            prefix_chars=0,
            token_count=10,
            start_page=3,
            end_page=3,
            spans=(SourceSpan(block_id=8, start=0, end=34),),
        ),
        Chunk(
            position=0,
            section_position=2,
            kind=ChunkKind.TABLE,
            text=header + "| South | 95 |",
            prefix_chars=19,
            token_count=10,
            start_page=3,
            end_page=3,
            spans=(SourceSpan(block_id=8, start=35, end=49),),
        ),
        Chunk(
            position=0,
            section_position=2,
            kind=ChunkKind.TABLE,
            text=header + "| East | 70 |",
            prefix_chars=19,
            token_count=10,
            start_page=3,
            end_page=3,
            spans=(SourceSpan(block_id=8, start=50, end=63),),
        ),
        Chunk(
            position=0,
            section_position=2,
            kind=ChunkKind.TABLE,
            text=header + "| West | 60 |",
            prefix_chars=19,
            token_count=10,
            start_page=3,
            end_page=3,
            spans=(SourceSpan(block_id=8, start=64, end=77),),
        ),
    ]
    assert result == expected


def test_table_chunks__rows_grouped_never_split() -> None:
    # Two rows fit the target; one row over the maximum still stays whole.
    table = "| Region | Sales |\n| North | 120 |\n| South | 95 |\n| A very long row with many words in it | 1 |"
    tested = helper_instance(helper_settings(15, 16, 1, 1))
    result = [
        (chunk.text, chunk.prefix_chars)
        for chunk in tested.table_chunks(0, helper_block(1, 1, table, BlockKind.TABLE), [])
    ]
    expected = [
        ("| Region | Sales |\n| North | 120 |\n| South | 95 |", 0),
        ("| Region | Sales |\n| A very long row with many words in it | 1 |", 19),
    ]
    assert result == expected


def test_table_chunks__the_caption_leads_every_chunk() -> None:
    table = "| Region | Sales |\n| North | 120 |\n| South | 95 |"
    caption = [helper_block(7, 3, "Table 2. Sales by region", BlockKind.PARAGRAPH)]
    tested = helper_instance(helper_settings(16, 20, 1, 1))
    result = tested.table_chunks(2, helper_block(8, 3, table, BlockKind.TABLE), caption)
    expected = [
        Chunk(
            position=0,
            section_position=2,
            kind=ChunkKind.TABLE,
            text="Table 2. Sales by region\n\n| Region | Sales |\n| North | 120 |",
            prefix_chars=0,
            token_count=16,
            start_page=3,
            end_page=3,
            spans=(SourceSpan(block_id=7, start=0, end=24), SourceSpan(block_id=8, start=0, end=34)),
        ),
        Chunk(
            position=0,
            section_position=2,
            kind=ChunkKind.TABLE,
            text="Table 2. Sales by region\n\n| Region | Sales |\n| South | 95 |",
            prefix_chars=45,
            token_count=16,
            start_page=3,
            end_page=3,
            spans=(SourceSpan(block_id=8, start=35, end=49),),
        ),
    ]
    assert result == expected


def test_chunk_of() -> None:
    group = PieceGroup(
        prefix=(helper_piece("Aa bb.", 3, 0, "")._replace(start_page=1, end_page=1),),
        pieces=(
            Piece(
                text="Cc dd.",
                token_count=3,
                spans=(SourceSpan(block_id=4, start=7, end=13),),
                start_page=2,
                end_page=2,
                paragraph=0,
                lead=" ",
            ),
            Piece(
                text="Ee ff.",
                token_count=3,
                spans=(SourceSpan(block_id=5, start=0, end=6),),
                start_page=3,
                end_page=3,
                paragraph=1,
                lead="",
            ),
        ),
    )
    tested = helper_instance(helper_settings(200, 350, 75, 1))
    result = tested.chunk_of(9, ChunkKind.PROSE, group)
    expected = Chunk(
        position=0,
        section_position=9,
        kind=ChunkKind.PROSE,
        text="Aa bb. Cc dd.\n\nEe ff.",
        prefix_chars=7,
        token_count=9,
        start_page=1,
        end_page=3,
        spans=(SourceSpan(block_id=4, start=7, end=13), SourceSpan(block_id=5, start=0, end=6)),
    )
    assert result == expected


def test_joined() -> None:
    pieces = (
        helper_piece("Aa.", 2, 0, ""),
        helper_piece("Bb.", 2, 0, "\n"),
        helper_piece("Cc.", 2, 1, ""),
        helper_piece("dd", 1, 1, ""),
    )
    tested = Chunker
    result = tested.joined(pieces)
    expected = "Aa.\nBb.\n\nCc.dd"
    assert result == expected


@pytest.mark.parametrize(
    "previous, piece, expected",
    [
        pytest.param(
            helper_piece("a", 1, 0, ""), helper_piece("b", 1, 0, "  "), "  ", id="same-paragraph-keeps-spacing"
        ),
        pytest.param(helper_piece("a", 1, 0, ""), helper_piece("b", 1, 0, ""), "", id="cut-without-space"),
        pytest.param(helper_piece("a", 1, 0, ""), helper_piece("b", 1, 1, ""), "\n\n", id="new-paragraph"),
    ],
)
def test_separator(previous: Piece, piece: Piece, expected: str) -> None:
    tested = Chunker
    result = tested.separator(previous, piece)
    assert result == expected


def test_merged_spans() -> None:
    spans = [
        SourceSpan(block_id=1, start=0, end=5),
        SourceSpan(block_id=1, start=6, end=9),
        SourceSpan(block_id=2, start=0, end=4),
        SourceSpan(block_id=1, start=12, end=14),
    ]
    tested = Chunker
    result = tested.merged_spans(spans)
    expected = (
        SourceSpan(block_id=1, start=0, end=9),
        SourceSpan(block_id=2, start=0, end=4),
        SourceSpan(block_id=1, start=12, end=14),
    )
    assert result == expected
