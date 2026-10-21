from __future__ import annotations

import pytest

from rag_my_docs.structures.piece import Piece
from rag_my_docs.structures.piece_group import PieceGroup
from rag_my_docs.structures.source_span import SourceSpan
from tests.conftest import is_namedtuple


def helper_piece(text: str, token_count: int) -> Piece:
    return Piece(
        text=text,
        token_count=token_count,
        spans=(SourceSpan(block_id=1, start=0, end=len(text)),),
        start_page=1,
        end_page=1,
        paragraph=0,
        lead="",
    )


def test_class() -> None:
    tested = PieceGroup
    fields = {"prefix": "tuple[Piece, ...]", "pieces": "tuple[Piece, ...]"}
    assert is_namedtuple(tested, fields)


@pytest.mark.parametrize(
    "tested, expected",
    [
        pytest.param(PieceGroup(prefix=(), pieces=()), 0, id="empty"),
        pytest.param(PieceGroup(prefix=(), pieces=(helper_piece("A b.", 3), helper_piece("C.", 2))), 5, id="own-only"),
        pytest.param(
            PieceGroup(prefix=(helper_piece("Before.", 2),), pieces=(helper_piece("A b.", 3),)), 5, id="with-prefix"
        ),
    ],
)
def test_token_count(tested: PieceGroup, expected: int) -> None:
    result = tested.token_count()
    assert result == expected


def test_to_dict() -> None:
    tested = PieceGroup(prefix=(helper_piece("Before.", 2),), pieces=(helper_piece("A b.", 3),))
    result = tested.to_dict()
    expected = {
        "prefix": [
            {
                "text": "Before.",
                "token_count": 2,
                "spans": [{"block_id": 1, "start": 0, "end": 7}],
                "start_page": 1,
                "end_page": 1,
                "paragraph": 0,
                "lead": "",
            }
        ],
        "pieces": [
            {
                "text": "A b.",
                "token_count": 3,
                "spans": [{"block_id": 1, "start": 0, "end": 4}],
                "start_page": 1,
                "end_page": 1,
                "paragraph": 0,
                "lead": "",
            }
        ],
    }
    assert result == expected


def test_from_dict() -> None:
    tested = PieceGroup
    result = tested.from_dict(
        {
            "prefix": [],
            "pieces": [
                {
                    "text": "A b.",
                    "token_count": 3,
                    "spans": [{"block_id": 1, "start": 0, "end": 4}],
                    "start_page": 1,
                    "end_page": 1,
                    "paragraph": 0,
                    "lead": "",
                }
            ],
        }
    )
    expected = PieceGroup(prefix=(), pieces=(helper_piece("A b.", 3),))
    assert result == expected
