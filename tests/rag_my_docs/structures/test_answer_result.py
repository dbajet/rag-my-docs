from __future__ import annotations

from typing import Any

from rag_my_docs.structures.answer_citation import AnswerCitation
from rag_my_docs.structures.answer_result import AnswerResult
from rag_my_docs.structures.model_answer import ModelAnswer
from rag_my_docs.structures.search_hit import SearchHit
from tests.conftest import is_namedtuple


def helper_instance() -> AnswerResult:
    return AnswerResult(
        question="What is the standard deduction?",
        passages=(SearchHit(1, 40, 7, 3, "Publication 501", 12, "Standard Deduction", 25, 25, "$15,750", 0, 0.78),),
        system="You answer questions.",
        prompt="[1] Publication 501 › Standard Deduction, page 25\n$15,750\n\nQuestion: What is the standard deduction?",
        answer=ModelAnswer(True, "$15,750.", (AnswerCitation(1, "Publication 501", 25),)),
        model="good",
        embedding_ms=12.5,
        search_ms=3.0,
        answer_ms=4200.0,
    )


def helper_dict() -> dict[str, Any]:
    return {
        "question": "What is the standard deduction?",
        "passages": [
            {
                "rank": 1,
                "chunk_id": 40,
                "chunk_position": 7,
                "document_id": 3,
                "document_name": "Publication 501",
                "section_id": 12,
                "section_path": "Standard Deduction",
                "start_page": 25,
                "end_page": 25,
                "text": "$15,750",
                "prefix_chars": 0,
                "similarity": 0.78,
            }
        ],
        "system": "You answer questions.",
        "prompt": "[1] Publication 501 › Standard Deduction, page 25\n$15,750\n\nQuestion: What is the standard deduction?",
        "answer": {
            "is_found": True,
            "text": "$15,750.",
            "citations": [{"passage": 1, "document": "Publication 501", "page": 25}],
        },
        "model": "good",
        "embedding_ms": 12.5,
        "search_ms": 3.0,
        "answer_ms": 4200.0,
    }


def test_class() -> None:
    tested = AnswerResult
    fields = {
        "question": "str",
        "passages": "tuple[SearchHit, ...]",
        "system": "str",
        "prompt": "str",
        "answer": "ModelAnswer",
        "model": "str",
        "embedding_ms": "float",
        "search_ms": "float",
        "answer_ms": "float",
    }
    assert is_namedtuple(tested, fields)


def test_to_dict() -> None:
    tested = helper_instance()
    result = tested.to_dict()
    expected = helper_dict()
    assert result == expected


def test_from_dict() -> None:
    tested = AnswerResult
    result = tested.from_dict(helper_dict())
    expected = helper_instance()
    assert result == expected
