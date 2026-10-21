from __future__ import annotations

from rag_my_docs.constants.block_kind import BlockKind
from rag_my_docs.constants.chunk_kind import ChunkKind
from rag_my_docs.constants.processing_constants import ProcessingConstants
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


class Chunker:
    """Groups neighbouring sentences into chunks, one section at a time, so no
    chunk ever mixes two sections. Within a section:

    - Paragraphs are split into sentences; a sentence longer than the maximum
      is cut at whitespace into pieces that fit (`TokenCounter.windows`).
    - Sentences are added in order until the target is reached. Near the
      target (and past the minimum), a new paragraph closes the chunk first;
      the maximum is never exceeded.
    - Each chunk after the first repeats the previous chunk's last sentence(s)
      as a prefix (the overlap), unless that would break the maximum.
    - A last chunk under the minimum joins the previous one when it fits.
    - A table is chunked alone, by whole rows, every chunk carrying the
      header row; a row is never split.
    - The short lines just before a table are its caption ("The following
      table lists…", "Table 5. Processor"): they leave the prose and lead every
      chunk of the table. Alone, a caption says what the table holds but not
      the answer, and a table says the answer without saying what it is about.

    A paragraph broken by a page break — no sentence end at the bottom of one
    page, a lowercase start at the top of the next — is joined back: chunks
    may cross pages, not sections. Every chunk keeps the block spans of its
    own text, so the original can be rebuilt and every chunk traced to its pages."""

    _OPEN_ENDINGS = '.!?:;"”’)'

    def __init__(self, settings: ChunkingSettings, splitter: SentenceSplitter, counter: TokenCounter) -> None:
        self._settings = settings
        self._splitter = splitter
        self._counter = counter

    def chunks(self, blocks: list[TextBlock], sections: list[Section]) -> list[Chunk]:
        by_id = {block.block_id: block for block in blocks}
        drafts: list[Chunk] = []
        for section in sections:
            drafts.extend(self.section_chunks(section.position, [by_id[block_id] for block_id in section.block_ids]))
        return [chunk._replace(position=position) for position, chunk in enumerate(drafts)]

    def section_chunks(self, section_position: int, blocks: list[TextBlock]) -> list[Chunk]:
        """Prose runs and tables in order; a prose run is always followed by a
        table, if anything, so its caption goes to the next run."""
        result: list[Chunk] = []
        runs = self.runs(blocks)
        caption: list[TextBlock] = []
        for index, run in enumerate(runs):
            if run[0].kind is BlockKind.TABLE:
                result.extend(self.table_chunks(section_position, run[0], caption))
                caption = []
                continue
            prose, caption = self.captioned(run) if index + 1 < len(runs) else (run, [])
            if prose:
                result.extend(self.prose_chunks(section_position, prose))
        return result

    @classmethod
    def runs(cls, blocks: list[TextBlock]) -> list[list[TextBlock]]:
        """Consecutive prose blocks together; each table alone."""
        result: list[list[TextBlock]] = []
        for block in blocks:
            if block.kind is BlockKind.TABLE or not result or result[-1][0].kind is BlockKind.TABLE:
                result.append([block])
            else:
                result[-1].append(block)
        return result

    def captioned(self, run: list[TextBlock]) -> tuple[list[TextBlock], list[TextBlock]]:
        """The prose, and the caption at its end: the last few short paragraphs,
        within the caption's limits."""
        start = len(run)
        tokens = 0
        for block in reversed(run):
            tokens += self._counter.count(block.text)
            if (
                block.kind is not BlockKind.PARAGRAPH
                or block.line_count > ProcessingConstants.caption_max_lines
                or len(run) - start >= ProcessingConstants.caption_max_blocks
                or tokens > ProcessingConstants.caption_max_tokens
            ):
                break
            start -= 1
        return run[:start], run[start:]

    def prose_chunks(self, section_position: int, blocks: list[TextBlock]) -> list[Chunk]:
        groups = self.merged_tail(self.grouped(self.pieces(self.paragraphs(blocks))))
        return [self.chunk_of(section_position, ChunkKind.PROSE, group) for group in groups]

    def paragraphs(self, blocks: list[TextBlock]) -> list[Paragraph]:
        result: list[Paragraph] = []
        previous: TextBlock | None = None
        for block in blocks:
            if previous is not None and self.continues(previous, block):
                last = result[-1]
                offset = len(last.text) + 1
                segment = ParagraphSegment(
                    block_id=block.block_id, page=block.page, offset=offset, length=len(block.text)
                )
                result[-1] = Paragraph(text=f"{last.text} {block.text}", segments=(*last.segments, segment))
            else:
                segment = ParagraphSegment(block_id=block.block_id, page=block.page, offset=0, length=len(block.text))
                result.append(Paragraph(text=block.text, segments=(segment,)))
            previous = block
        return result

    @classmethod
    def continues(cls, previous: TextBlock, block: TextBlock) -> bool:
        return (
            block.page == previous.page + 1
            and not previous.text.endswith(tuple(cls._OPEN_ENDINGS))
            and block.text[0].islower()
        )

    def pieces(self, paragraphs: list[Paragraph]) -> list[Piece]:
        result: list[Piece] = []
        for index, paragraph in enumerate(paragraphs):
            previous_end: int | None = None
            for sentence in self._splitter.sentences(paragraph.text):
                for window in self.fitted(paragraph.text, sentence.start, sentence.end):
                    lead = "" if previous_end is None else paragraph.text[previous_end : window.start]
                    result.append(self.piece(paragraph, index, window.start, window.end, lead))
                    previous_end = window.end
        return result

    def fitted(self, text: str, start: int, end: int) -> list[TextRange]:
        """The sentence as one range, or cut into ranges of at most the maximum."""
        sentence = text[start:end]
        if self._counter.count(sentence) <= self._settings.max_tokens:
            return [TextRange(start=start, end=end)]
        return [
            TextRange(start=start + window.start, end=start + window.end)
            for window in self._counter.windows(sentence, self._settings.max_tokens)
        ]

    def piece(self, paragraph: Paragraph, index: int, start: int, end: int, lead: str) -> Piece:
        text = paragraph.text[start:end]
        first_page, last_page = paragraph.pages(start, end)
        return Piece(
            text=text,
            token_count=self._counter.count(text),
            spans=paragraph.spans(start, end),
            start_page=first_page,
            end_page=last_page,
            paragraph=index,
            lead=lead,
        )

    def grouped(self, pieces: list[Piece]) -> list[PieceGroup]:
        result: list[PieceGroup] = []
        current = PieceGroup(prefix=(), pieces=())
        for piece in pieces:
            if current.pieces and self.is_full(current, piece):
                result.append(current)
                current = self.next_group(current, piece)
            current = current._replace(pieces=(*current.pieces, piece))
        if current.pieces:
            result.append(current)
        return result

    def is_full(self, group: PieceGroup, piece: Piece) -> bool:
        tokens = group.token_count()
        combined = tokens + piece.token_count
        settings = self._settings
        if combined > settings.max_tokens or tokens >= settings.target_tokens:
            return True
        is_new_paragraph = piece.paragraph != group.pieces[-1].paragraph
        return is_new_paragraph and tokens >= settings.min_tokens and combined > settings.target_tokens

    def next_group(self, closed: PieceGroup, piece: Piece) -> PieceGroup:
        """Starts with the closed chunk's last sentences, when they fit with the next piece."""
        count = self._settings.overlap_sentences
        prefix = closed.pieces[-count:] if count else ()
        if sum(item.token_count for item in prefix) + piece.token_count > self._settings.max_tokens:
            prefix = ()
        return PieceGroup(prefix=tuple(prefix), pieces=())

    def merged_tail(self, groups: list[PieceGroup]) -> list[PieceGroup]:
        if len(groups) < 2:
            return groups
        last = groups[-1]
        before = groups[-2]
        own = sum(piece.token_count for piece in last.pieces)
        if own >= self._settings.min_tokens or before.token_count() + own > self._settings.max_tokens:
            return groups
        return [*groups[:-2], before._replace(pieces=(*before.pieces, *last.pieces))]

    def table_chunks(self, section_position: int, block: TextBlock, caption: list[TextBlock]) -> list[Chunk]:
        """Whole rows, grouped up to the target; the caption and the header row
        are the first chunk's own text and every later chunk's prefix."""
        leads = [
            Piece(
                text=line.text,
                token_count=self._counter.count(line.text),
                spans=(SourceSpan(block_id=line.block_id, start=0, end=len(line.text)),),
                start_page=line.page,
                end_page=line.page,
                paragraph=index,
                lead="",
            )
            for index, line in enumerate(caption)
        ]
        pieces: list[Piece] = []
        start = 0
        for line in block.text.split("\n"):
            span = SourceSpan(block_id=block.block_id, start=start, end=start + len(line))
            pieces.append(
                Piece(
                    text=line,
                    token_count=self._counter.count(line),
                    spans=(span,),
                    start_page=block.page,
                    end_page=block.page,
                    paragraph=len(caption),
                    lead="" if start == 0 else "\n",
                )
            )
            start += len(line) + 1
        heading = (*leads, pieces[0])
        groups: list[PieceGroup] = []
        current = PieceGroup(prefix=(), pieces=heading)
        for row in pieces[1:]:
            tokens = current.token_count()
            # The first chunk owns the caption and header; a chunk is never closed before its first row.
            has_rows = len(current.pieces) > (0 if current.prefix else len(heading))
            if has_rows and (
                tokens + row.token_count > self._settings.max_tokens or tokens >= self._settings.target_tokens
            ):
                groups.append(current)
                current = PieceGroup(prefix=heading, pieces=())
            current = current._replace(pieces=(*current.pieces, row))
        groups.append(current)
        return [self.chunk_of(section_position, ChunkKind.TABLE, group) for group in groups]

    def chunk_of(self, section_position: int, kind: ChunkKind, group: PieceGroup) -> Chunk:
        own = self.joined(group.pieces)
        prefix = ""
        if group.prefix:
            prefix = self.joined(group.prefix) + self.separator(group.prefix[-1], group.pieces[0])
        text = prefix + own
        everything = (*group.prefix, *group.pieces)
        return Chunk(
            position=0,
            section_position=section_position,
            kind=kind,
            text=text,
            prefix_chars=len(prefix),
            token_count=self._counter.count(text),
            start_page=min(piece.start_page for piece in everything),
            end_page=max(piece.end_page for piece in everything),
            spans=self.merged_spans([span for piece in group.pieces for span in piece.spans]),
        )

    @classmethod
    def joined(cls, pieces: tuple[Piece, ...]) -> str:
        result = pieces[0].text
        for previous, piece in zip(pieces, pieces[1:], strict=False):
            result += cls.separator(previous, piece) + piece.text
        return result

    @classmethod
    def separator(cls, previous: Piece, piece: Piece) -> str:
        """The original spacing inside a paragraph; a blank line between paragraphs."""
        return piece.lead if piece.paragraph == previous.paragraph else "\n\n"

    @classmethod
    def merged_spans(cls, spans: list[SourceSpan]) -> tuple[SourceSpan, ...]:
        """Neighbouring spans of one block as one: the whitespace between them is the block's own."""
        result: list[SourceSpan] = []
        for span in spans:
            if result and result[-1].block_id == span.block_id:
                result[-1] = result[-1]._replace(end=span.end)
            else:
                result.append(span)
        return tuple(result)
