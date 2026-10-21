from __future__ import annotations

from rag_my_docs.structures.piece import Piece
from rag_my_docs.structures.source_span import SourceSpan
from tests.conftest import is_namedtuple


def helper_instance() -> Piece:
    return Piece(
        text="It rose 3.5 percent.",
        token_count=7,
        spans=(SourceSpan(block_id=2, start=12, end=32),),
        start_page=1,
        end_page=1,
        paragraph=0,
        lead=" ",
    )


def test_class() -> None:
    tested = Piece
    fields = {
        "text": "str",
        "token_count": "int",
        "spans": "tuple[SourceSpan, ...]",
        "start_page": "int",
        "end_page": "int",
        "paragraph": "int",
        "lead": "str",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = helper_instance()
    result = tested.to_dict()
    expected = {
        "text": "It rose 3.5 percent.",
        "token_count": 7,
        "spans": [{"block_id": 2, "start": 12, "end": 32}],
        "start_page": 1,
        "end_page": 1,
        "paragraph": 0,
        "lead": " ",
    }
    assert result == expected


def test_from_dict() -> None:
    tested = Piece
    result = tested.from_dict(
        {
            "text": "It rose 3.5 percent.",
            "token_count": "7",
            "spans": [{"block_id": "2", "start": "12", "end": "32"}],
            "start_page": "1",
            "end_page": 1,
            "paragraph": "0",
            "lead": " ",
        }
    )
    expected = helper_instance()
    assert result == expected
