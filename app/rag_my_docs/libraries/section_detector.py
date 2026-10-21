from __future__ import annotations

import re
from collections import Counter

from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.constants.processing_constants import ProcessingConstants
from rag_my_docs.constants.section_source import SectionSource
from rag_my_docs.structures.heading import Heading
from rag_my_docs.structures.numbering import Numbering
from rag_my_docs.structures.outline_entry import OutlineEntry
from rag_my_docs.structures.section import Section
from rag_my_docs.structures.text_block import TextBlock


class SectionDetector:
    """Decides where sections start, trying four strategies in order and
    keeping the first that works:

    1. Bookmarks, when most of them are found as heading text near the page
       they point to. A bookmark whose title is not found still starts a
       section, at the top of its page: bookmarks name pages, not lines.
    2. Numbered headings ("2.1 Scope", "Section 3", "Article IV"), when they
       stand out — bold, or a slightly larger font — or are a keyword heading
       alone on its line. A plain "1. Buy milk" in body font is a list item.
    3. Layout: short lines in a clearly larger font, or a bold line alone
       before regular text. Bold words inside a paragraph never qualify.
    4. Otherwise one untitled section holds the whole document: no heading is
       invented. Paragraphs still guide the chunks inside it.

    Strategies 2 and 3 need at least two headings to be trusted. Only short
    blocks qualify: at most two lines, no sentence punctuation at the end, not
    starting in lowercase, not a page header or footer."""

    _DOTTED = re.compile(r"^(?P<number>\d{1,2}(?:\.\d{1,2})*)\.?\s+(?P<title>\S.*)$")
    _KEYWORD = re.compile(
        r"^(?P<word>part|chapter|section|article|appendix|annex)\s+"
        r"(?P<number>\d{1,3}(?:\.\d{1,3})*|[ivxlcdm]{1,7}|[a-z])\b\.?:?\s*(?P<title>.*)$",
        re.IGNORECASE,
    )
    _RANKS = {"part": 0, "chapter": 1, "section": 2, "article": 2, "appendix": 2, "annex": 2}
    _BARE_RANK = 2
    _ENDINGS = (".", ",", ";", ":")
    _NOT_TITLE = re.compile(r"[\W_]+")

    def detected(
        self, blocks: list[TextBlock], outline: tuple[OutlineEntry, ...]
    ) -> tuple[list[TextBlock], list[Section]]:
        """The blocks, headings marked as such, and the sections they open."""
        body = self.body_size(blocks)
        headings = self.from_bookmarks(blocks, outline) or self.numbered(blocks, body) or self.by_layout(blocks, body)
        heading_ids = {heading.block_id for heading in headings if heading.block_id is not None}
        marked = [
            block._replace(kind=BlockKind.HEADING) if block.block_id in heading_ids else block for block in blocks
        ]
        return marked, self.sections_of(marked, headings)

    @classmethod
    def body_size(cls, blocks: list[TextBlock]) -> float:
        """The font size of most of the paragraph text; 0 for none."""
        sizes: Counter[float] = Counter()
        for block in blocks:
            if block.kind is BlockKind.PARAGRAPH:
                sizes[block.font_size] += len(block.text)
        return sizes.most_common(1)[0][0] if sizes else 0.0

    @classmethod
    def is_candidate(cls, block: TextBlock) -> bool:
        text = block.text
        return (
            block.kind is BlockKind.PARAGRAPH
            and block.line_count <= ProcessingConstants.heading_max_lines
            and len(text) <= ProcessingConstants.heading_max_chars
            and not text.endswith(cls._ENDINGS)
            and not text[0].islower()
            and any(character.isalpha() for character in text)
        )

    @classmethod
    def numbering(cls, text: str) -> Numbering | None:
        if found := cls._KEYWORD.match(text):
            number = found.group("number")
            return Numbering(
                number=f"{found.group('word')} {number}",
                title=found.group("title"),
                rank=cls._RANKS[found.group("word").lower()],
                depth=number.count(".") + 1,
                is_keyword=True,
            )
        if (found := cls._DOTTED.match(text)) and not found.group("title")[0].islower():
            number = found.group("number")
            return Numbering(
                number=number,
                title=found.group("title"),
                rank=cls._BARE_RANK,
                depth=number.count(".") + 1,
                is_keyword=False,
            )
        return None

    def from_bookmarks(self, blocks: list[TextBlock], outline: tuple[OutlineEntry, ...]) -> list[Heading]:
        result: list[Heading] = []
        found = 0
        cursor = 0
        for entry in outline:
            if (match := self.bookmark_block(blocks, entry, cursor)) is not None:
                found += 1
                cursor = match.block_id + 1
                numbering = self.numbering(match.text)
                result.append(
                    Heading(
                        block_id=match.block_id,
                        start_block_id=match.block_id,
                        level=entry.level,
                        number="" if numbering is None else numbering.number,
                        title=match.text,
                        source=SectionSource.BOOKMARK,
                    )
                )
            elif (start := self.first_block_on(blocks, entry.page, cursor)) is not None:
                cursor = start.block_id
                result.append(
                    Heading(
                        block_id=None,
                        start_block_id=start.block_id,
                        level=entry.level,
                        number="",
                        title=entry.title,
                        source=SectionSource.BOOKMARK,
                    )
                )
        if not outline or found / len(outline) < ProcessingConstants.bookmark_match_share:
            return []
        return result

    def bookmark_block(self, blocks: list[TextBlock], entry: OutlineEntry, cursor: int) -> TextBlock | None:
        """The block on the bookmark's page, or the next, whose text is its title."""
        title = self.bare(entry.title)
        for block in blocks[cursor:]:
            if block.page > entry.page + 1:
                return None
            if block.page >= entry.page and self.is_candidate(block) and self.bare(block.text) == title:
                return block
        return None

    @classmethod
    def first_block_on(cls, blocks: list[TextBlock], page: int, cursor: int) -> TextBlock | None:
        for block in blocks[cursor:]:
            if block.page == page and block.kind not in (BlockKind.HEADER, BlockKind.FOOTER):
                return block
            if block.page > page:
                return None
        return None

    @classmethod
    def bare(cls, text: str) -> str:
        """A title as compared: without its number, case, spacing or punctuation."""
        numbering = cls.numbering(text)
        title = text if numbering is None or not numbering.title else numbering.title
        return cls._NOT_TITLE.sub(" ", title).strip().casefold()

    def numbered(self, blocks: list[TextBlock], body: float) -> list[Heading]:
        result: list[Heading] = []
        for block in blocks:
            if not self.is_candidate(block) or (numbering := self.numbering(block.text)) is None:
                continue
            is_prominent = (
                block.bold_share >= ProcessingConstants.bold_share
                or block.font_size >= body * ProcessingConstants.numbered_size_ratio
            )
            if is_prominent or (numbering.is_keyword and block.line_count == 1):
                result.append(
                    Heading(
                        block_id=block.block_id,
                        start_block_id=block.block_id,
                        level=numbering.level(),
                        number=numbering.number,
                        title=block.text,
                        source=SectionSource.NUMBERED,
                    )
                )
        return result if len(result) >= ProcessingConstants.min_headings else []

    def by_layout(self, blocks: list[TextBlock], body: float) -> list[Heading]:
        """Larger fonts make levels, largest first; bold body-size lines come after them."""
        large = [
            block
            for block in blocks
            if self.is_candidate(block) and block.font_size >= body * ProcessingConstants.heading_size_ratio
        ]
        large_ids = {block.block_id for block in large}
        bold = [
            block
            for index, block in enumerate(blocks)
            if block.block_id not in large_ids and self.is_bold_label(blocks, index, body)
        ]
        steps = sorted({self.size_step(block.font_size) for block in large}, reverse=True)
        found = [(block, steps.index(self.size_step(block.font_size)) + 1) for block in large]
        found += [(block, len(steps) + 1) for block in bold]
        result = [
            Heading(
                block_id=block.block_id,
                start_block_id=block.block_id,
                level=level,
                number="",
                title=block.text,
                source=SectionSource.LAYOUT,
            )
            for block, level in sorted(found, key=lambda pair: pair[0].block_id)
        ]
        return result if len(result) >= ProcessingConstants.min_headings else []

    def is_bold_label(self, blocks: list[TextBlock], index: int, body: float) -> bool:
        """A whole bold line, not smaller than the text, followed by regular text."""
        block = blocks[index]
        following = next(
            (later for later in blocks[index + 1 :] if later.kind not in (BlockKind.HEADER, BlockKind.FOOTER)),
            None,
        )
        return (
            self.is_candidate(block)
            and block.line_count == 1
            and block.bold_share >= ProcessingConstants.bold_share
            and block.font_size >= body - ProcessingConstants.size_step
            and following is not None
            and following.kind is BlockKind.PARAGRAPH
            and following.bold_share < ProcessingConstants.bold_share
        )

    @classmethod
    def size_step(cls, size: float) -> float:
        return round(size / ProcessingConstants.size_step) * ProcessingConstants.size_step

    def sections_of(self, blocks: list[TextBlock], headings: list[Heading]) -> list[Section]:
        """Each heading opens a section at its start block; a section's parent is
        the nearest earlier one of a higher level. Text before the first heading
        goes to an untitled section; page headers and footers belong to none."""
        levels = {level: rank + 1 for rank, level in enumerate(sorted({heading.level for heading in headings}))}
        starts: dict[int, list[Heading]] = {}
        for heading in headings:
            starts.setdefault(heading.start_block_id, []).append(heading)
        result: list[Section] = []
        open_sections: list[Section] = []
        for block in blocks:
            for heading in starts.get(block.block_id, []):
                level = levels[heading.level]
                while open_sections and open_sections[-1].level >= level:
                    open_sections.pop()
                parent = open_sections[-1].position if open_sections else None
                section = self.opened(len(result), parent, level, heading, block.page)
                result.append(section)
                open_sections.append(section)
            if block.kind in (BlockKind.HEADER, BlockKind.FOOTER) or block.kind is BlockKind.HEADING:
                continue
            if not result:
                result.append(self.untitled(block.page))
            current = result[-1]
            result[-1] = current._replace(
                block_ids=(*current.block_ids, block.block_id),
                start_page=min(current.start_page, block.page),
                end_page=max(current.end_page, block.page),
            )
        return self.spread(result)

    @classmethod
    def opened(cls, position: int, parent: int | None, level: int, heading: Heading, page: int) -> Section:
        return Section(
            position=position,
            parent_position=parent,
            level=level,
            number=heading.number,
            title=heading.title,
            source=heading.source,
            heading_block_id=heading.block_id,
            block_ids=(),
            start_page=page,
            end_page=page,
        )

    @classmethod
    def untitled(cls, page: int) -> Section:
        return Section(
            position=0,
            parent_position=None,
            level=1,
            number="",
            title="",
            source=SectionSource.FALLBACK,
            heading_block_id=None,
            block_ids=(),
            start_page=page,
            end_page=page,
        )

    @classmethod
    def spread(cls, sections: list[Section]) -> list[Section]:
        """A section's pages run to its last subsection's last page."""
        result = list(sections)
        for section in reversed(result):
            if section.parent_position is not None:
                parent = result[section.parent_position]
                result[section.parent_position] = parent._replace(
                    start_page=min(parent.start_page, section.start_page),
                    end_page=max(parent.end_page, section.end_page),
                )
        return result
