from __future__ import annotations

import dataclasses

from rag_my_docs.constants.document_constants import DocumentConstants
from tests.conftest import is_dataclass


def test_class() -> None:
    tested = DocumentConstants
    fields = {"download_cache_control": "str", "not_found": "str"}
    assert is_dataclass(tested, fields)


def test_values() -> None:
    tested = DocumentConstants
    result = dataclasses.asdict(tested)
    expected = {"download_cache_control": "private, no-store", "not_found": "This document does not exist."}
    assert result == expected
