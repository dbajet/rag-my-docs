from __future__ import annotations

from unittest.mock import call, patch

import pytest

from rag_my_docs.commands.answer_command import AnswerCommand
from rag_my_docs.commands.evaluation_command import EvaluationCommand
from rag_my_docs.libraries.claude_box import ClaudeBox
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.embedding_box import EmbeddingBox
from rag_my_docs.libraries.llm_connection_store import LlmConnectionStore
from rag_my_docs.libraries.llm_gateway import LlmGateway
from rag_my_docs.libraries.openai_chat_box import OpenAiChatBox
from rag_my_docs.libraries.search_store import SearchStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.structures.answer_result import AnswerResult
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.evaluation_case import EvaluationCase
from rag_my_docs.structures.evaluation_outcome import EvaluationOutcome
from rag_my_docs.structures.model_answer import ModelAnswer
from rag_my_docs.structures.search_hit import SearchHit
from rag_my_docs.structures.user_account import UserAccount

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="


def helper_answers() -> AnswerCommand:
    database = Database("postgresql://u:p@h/d")
    sealer = SecretSealer(KEY)
    return AnswerCommand(
        LlmGateway(LlmConnectionStore(database, sealer), OpenAiChatBox(), ClaudeBox()),
        SearchStore(database, sealer, SectionPaths(database, sealer)),
        EmbeddingBox(EmbeddingSettings(model="BAAI/bge-small-en-v1.5", query_prefix="Q: ", batch_size=32)),
    )


def helper_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase("Standard deduction?", ("Publication 501", "Publication 17"), ("15,750",), True),
        EvaluationCase("Passport?", (), (), False),
    ]


def helper_user() -> UserAccount:
    return UserAccount(user_id=7, email="owner@example.com", display_name="Owner", is_disabled=False)


def helper_result(documents: list[str], is_found: bool, text: str) -> AnswerResult:
    passages = tuple(
        SearchHit(rank, rank, 0, rank, name, 1, "Section", 1, 1, "Text", 0, 0.8)
        for rank, name in enumerate(documents, start=1)
    )
    return AnswerResult("Q?", passages, "rules", "prompt", ModelAnswer(is_found, text, ()), "good", 1.0, 2.0, 3.0)


def test___init__() -> None:
    answers = helper_answers()
    cases = helper_cases()
    tested = EvaluationCommand(answers, cases)
    result = (tested._answers, tested._cases)
    expected = (answers, cases)
    assert result == expected


def test_cases() -> None:
    tested = EvaluationCommand(helper_answers(), helper_cases())
    result = tested.cases()
    expected = helper_cases()
    assert result == expected


def test_evaluated() -> None:
    tested = EvaluationCommand(helper_answers(), helper_cases())
    answered = helper_result(["Publication 17"], True, "It is $15,750.")
    scored = EvaluationOutcome("Q", (), (), True, True, answered)
    with (
        patch.object(AnswerCommand, "answered", side_effect=[answered]) as mock_answered,
        patch.object(EvaluationCommand, "scored", side_effect=[scored]) as mock_scored,
    ):
        result = tested.evaluated(helper_user(), 0, 10)
    expected = EvaluationOutcome("Q", (), (), True, True, answered)
    assert result == expected
    exp_calls = [call(helper_user(), "Standard deduction?", 10)]
    assert mock_answered.mock_calls == exp_calls
    exp_calls = [call(helper_cases()[0], answered)]
    assert mock_scored.mock_calls == exp_calls


@pytest.mark.parametrize("index", [pytest.param(-1, id="before"), pytest.param(2, id="after")])
def test_evaluated__out_of_range(index: int) -> None:
    tested = EvaluationCommand(helper_answers(), helper_cases())
    with (
        patch.object(AnswerCommand, "answered", side_effect=[]) as mock_answered,
        pytest.raises(AppException) as exc_info,
    ):
        tested.evaluated(helper_user(), index, 5)
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (404, "This test question does not exist.")
    assert result == expected
    exp_calls = []
    assert mock_answered.mock_calls == exp_calls


@pytest.mark.parametrize(
    "case, documents, is_found, text, expected",
    [
        pytest.param(
            EvaluationCase("Q", ("Publication 501", "Publication 17"), ("15,750", "single|alone"), True),
            ["Publication 334", "Publication 17"],
            True,
            "SINGLE: $15,750.",
            ((), (), True, True),
            id="any-document-and-alternative-fact",
        ),
        pytest.param(
            EvaluationCase("Q", ("Publication 501",), ("15,750", "single|alone"), True),
            ["Publication 334"],
            True,
            "It is $15,750.",
            (("Publication 501",), ("single|alone",), False, False),
            id="no-passage-from-it-and-a-fact-missing",
        ),
        pytest.param(
            EvaluationCase("Q", ("Publication 501",), ("15,750",), True),
            ["Publication 501"],
            False,
            "15,750 is not stated.",
            ((), (), True, False),
            id="facts-but-said-not-found",
        ),
        pytest.param(
            EvaluationCase("Passport?", (), (), False),
            ["Publication 17"],
            False,
            "Not in the publications.",
            ((), (), True, True),
            id="unanswerable-said-so",
        ),
        pytest.param(
            EvaluationCase("Passport?", (), (), False),
            [],
            True,
            "Go to the post office.",
            ((), (), True, False),
            id="unanswerable-answered-anyway",
        ),
    ],
)
def test_scored(
    case: EvaluationCase,
    documents: list[str],
    is_found: bool,
    text: str,
    expected: tuple[tuple[str, ...], tuple[str, ...], bool, bool],
) -> None:
    tested = EvaluationCommand
    answered = helper_result(documents, is_found, text)
    outcome = tested.scored(case, answered)
    result = (outcome.missing_documents, outcome.missing_facts, outcome.is_retrieval_right, outcome.is_answer_right)
    assert result == expected
    exp_whole = EvaluationOutcome(case.question, *expected, answered)
    assert outcome == exp_whole
