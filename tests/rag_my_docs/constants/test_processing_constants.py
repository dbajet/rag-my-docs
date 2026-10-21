from __future__ import annotations

import dataclasses

from rag_my_docs.constants.processing_constants import ProcessingConstants
from tests.conftest import is_dataclass


def test_class() -> None:
    tested = ProcessingConstants
    fields = {
        "env_target_tokens": "str",
        "env_max_tokens": "str",
        "env_min_tokens": "str",
        "env_overlap_sentences": "str",
        "default_target_tokens": "int",
        "default_max_tokens": "int",
        "default_min_tokens": "int",
        "default_overlap_sentences": "int",
        "strategy_name": "str",
        "furniture_margin": "float",
        "furniture_page_share": "float",
        "furniture_min_pages": "int",
        "column_tolerance": "float",
        "table_overlap": "float",
        "caption_max_blocks": "int",
        "caption_max_lines": "int",
        "caption_max_tokens": "int",
        "bold_share": "float",
        "heading_max_chars": "int",
        "heading_max_lines": "int",
        "heading_size_ratio": "float",
        "numbered_size_ratio": "float",
        "size_step": "float",
        "min_headings": "int",
        "bookmark_match_share": "float",
        "unreadable": "str",
        "protected": "str",
        "unexpected": "str",
        "not_pdf": "str",
        "not_processed": "str",
        "no_text": "str",
        "busy": "str",
        "stale_run_minutes": "int",
    }
    assert is_dataclass(tested, fields)


def test_values() -> None:
    tested = ProcessingConstants
    result = dataclasses.asdict(tested)
    expected = {
        "env_target_tokens": "CHUNK_TARGET_TOKENS",
        "env_max_tokens": "CHUNK_MAX_TOKENS",
        "env_min_tokens": "CHUNK_MIN_TOKENS",
        "env_overlap_sentences": "CHUNK_OVERLAP_SENTENCES",
        "default_target_tokens": 200,
        "default_max_tokens": 350,
        "default_min_tokens": 75,
        "default_overlap_sentences": 1,
        "strategy_name": "sentences-v2",
        "furniture_margin": 0.1,
        "furniture_page_share": 0.5,
        "furniture_min_pages": 2,
        "column_tolerance": 0.05,
        "table_overlap": 0.5,
        "caption_max_blocks": 2,
        "caption_max_lines": 2,
        "caption_max_tokens": 60,
        "bold_share": 0.9,
        "heading_max_chars": 120,
        "heading_max_lines": 2,
        "heading_size_ratio": 1.15,
        "numbered_size_ratio": 1.05,
        "size_step": 0.5,
        "min_headings": 2,
        "bookmark_match_share": 0.6,
        "unreadable": "This PDF could not be read: the file is damaged or is not a PDF.",
        "protected": "This PDF is protected by a password.",
        "unexpected": "Processing failed unexpectedly. Try again, or report it.",
        "not_pdf": "Only PDF files are processed for search for now.",
        "not_processed": "This document has not been processed yet.",
        "no_text": "No page of this PDF has extractable text; it is probably scanned (OCR is not supported yet).",
        "busy": "This document is already being processed.",
        "stale_run_minutes": 10,
    }
    assert result == expected
