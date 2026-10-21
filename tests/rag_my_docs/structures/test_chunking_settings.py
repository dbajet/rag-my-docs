from __future__ import annotations

import pytest

from rag_my_docs.structures.chunking_settings import ChunkingSettings
from tests.conftest import is_namedtuple


def test_class() -> None:
    tested = ChunkingSettings
    fields = {"target_tokens": "int", "max_tokens": "int", "min_tokens": "int", "overlap_sentences": "int"}
    assert is_namedtuple(tested, fields)


@pytest.mark.parametrize(
    "tested, expected",
    [
        pytest.param(
            ChunkingSettings(target_tokens=200, max_tokens=350, min_tokens=75, overlap_sentences=1),
            "sentences-v2 target=200 max=350 min=75 overlap=1",
            id="defaults",
        ),
        pytest.param(
            ChunkingSettings(target_tokens=120, max_tokens=180, min_tokens=40, overlap_sentences=0),
            "sentences-v2 target=120 max=180 min=40 overlap=0",
            id="no-overlap",
        ),
    ],
)
def test_strategy(tested: ChunkingSettings, expected: str) -> None:
    result = tested.strategy()
    assert result == expected


def test_to_dict() -> None:
    tested = ChunkingSettings(target_tokens=200, max_tokens=350, min_tokens=75, overlap_sentences=1)
    result = tested.to_dict()
    expected = {"target_tokens": 200, "max_tokens": 350, "min_tokens": 75, "overlap_sentences": 1}
    assert result == expected


def test_from_dict() -> None:
    tested = ChunkingSettings
    result = tested.from_dict({"target_tokens": "120", "max_tokens": 180, "min_tokens": "40", "overlap_sentences": "2"})
    expected = ChunkingSettings(target_tokens=120, max_tokens=180, min_tokens=40, overlap_sentences=2)
    assert result == expected
