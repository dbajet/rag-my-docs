from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _ProcessingConstants:
    # --- chunking defaults (each overridable from the environment) -------------
    env_target_tokens: str = "CHUNK_TARGET_TOKENS"
    env_max_tokens: str = "CHUNK_MAX_TOKENS"
    env_min_tokens: str = "CHUNK_MIN_TOKENS"
    env_overlap_sentences: str = "CHUNK_OVERLAP_SENTENCES"
    default_target_tokens: int = 200
    default_max_tokens: int = 350
    default_min_tokens: int = 75
    default_overlap_sentences: int = 1
    # Named in every chunk, so chunks made under other settings can be told apart.
    # v2: printed page numbers make a footer; a table's caption joins its chunks.
    strategy_name: str = "sentences-v2"

    # --- layout ------------------------------------------------------------------
    # The top and bottom share of a page where running headers and footers live.
    furniture_margin: float = 0.1
    # A margin line is a header or footer when it repeats on at least this share
    # of the pages that have text, and on at least two of them.
    furniture_page_share: float = 0.5
    furniture_min_pages: int = 2
    # A block crossing the page's middle by more than this share of its width spans both columns.
    column_tolerance: float = 0.05
    # A block covered at least this much by a table is the table's own text, not prose.
    table_overlap: float = 0.5
    # The short lines just before a table ("The following table lists…",
    # "Table 5. Processor") are its caption and are chunked with it: at most
    # this many blocks, each of at most this many lines, together under this many tokens.
    caption_max_blocks: int = 2
    caption_max_lines: int = 2
    caption_max_tokens: int = 60
    # A block is bold when at least this share of its characters is bold.
    bold_share: float = 0.9

    # --- sections ------------------------------------------------------------------
    heading_max_chars: int = 120
    heading_max_lines: int = 2
    # A font this much larger than the body text marks a heading by size alone.
    heading_size_ratio: float = 1.15
    # A numbered heading needs only a slightly larger font, or bold.
    numbered_size_ratio: float = 1.05
    # Sizes closer than this are the same heading level.
    size_step: float = 0.5
    # A strategy is used only when it finds at least this many headings.
    min_headings: int = 2
    # Bookmarks are trusted when at least this share of them is found in the text.
    bookmark_match_share: float = 0.6

    # --- messages --------------------------------------------------------------------
    unreadable: str = "This PDF could not be read: the file is damaged or is not a PDF."
    protected: str = "This PDF is protected by a password."
    unexpected: str = "Processing failed unexpectedly. Try again, or report it."
    not_pdf: str = "Only PDF files are processed for search for now."
    not_processed: str = "This document has not been processed yet."
    no_text: str = "No page of this PDF has extractable text; it is probably scanned (OCR is not supported yet)."
    busy: str = "This document is already being processed."
    # A run that started this long ago on a color that is gone is started again.
    stale_run_minutes: int = 10


ProcessingConstants = _ProcessingConstants()
