from __future__ import annotations

from rag_my_docs.structures.raw_table import RawTable
from tests.conftest import is_namedtuple


def helper_instance() -> RawTable:
    return RawTable(
        page=2,
        x0=72.0,
        y0=420.0,
        x1=522.0,
        y1=480.0,
        header=("Region", "Sales"),
        rows=(("North", "120"), ("South", "")),
    )


def test_class() -> None:
    tested = RawTable
    fields = {
        "page": "int",
        "x0": "float",
        "y0": "float",
        "x1": "float",
        "y1": "float",
        "header": "tuple[str, ...]",
        "rows": "tuple[tuple[str, ...], ...]",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = helper_instance()
    result = tested.to_dict()
    expected = {
        "page": 2,
        "x0": 72.0,
        "y0": 420.0,
        "x1": 522.0,
        "y1": 480.0,
        "header": ["Region", "Sales"],
        "rows": [["North", "120"], ["South", ""]],
    }
    assert result == expected


def test_from_dict() -> None:
    tested = RawTable
    result = tested.from_dict(
        {
            "page": "2",
            "x0": 72,
            "y0": 420,
            "x1": "522",
            "y1": 480,
            "header": ["Region", "Sales"],
            "rows": [["North", "120"], ["South", ""]],
        }
    )
    expected = helper_instance()
    assert result == expected
