from __future__ import annotations

from typing import Any
from unittest.mock import call, patch

import pytest

from rag_my_docs.commands.answer_command import AnswerCommand
from rag_my_docs.libraries.claude_box import ClaudeBox
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.embedding_box import EmbeddingBox
from rag_my_docs.libraries.llm_connection_store import LlmConnectionStore
from rag_my_docs.libraries.llm_gateway import LlmGateway
from rag_my_docs.libraries.openai_chat_box import OpenAiChatBox
from rag_my_docs.libraries.search_store import SearchStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.structures.answer_citation import AnswerCitation
from rag_my_docs.structures.answer_result import AnswerResult
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.model_answer import ModelAnswer
from rag_my_docs.structures.search_hit import SearchHit
from rag_my_docs.structures.user_account import UserAccount

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="
SPACE = "BAAI/bge-small-en-v1.5+context-v3"
SYSTEM = (
    "You answer questions about U.S. federal income tax for tax year 2025.\n"
    "Use only the numbered passages below, taken from IRS publications. Use nothing you know from elsewhere.\n"
    "If the passages do not contain the answer, say so plainly and set found to false.\n"
    "Answer briefly, in the language of the question, with the exact figures and words of the passages.\n"
    'Reply with JSON only: {"found": true or false, "answer": "...", "passages": [the numbers of the passages\n'
    "the answer comes from]}."
)
PROMPT = (
    "[1] Publication 501 › Standard Deduction, page 25\n"
    "Single $15,750\n"
    "\n"
    "[2] Publication 17 › Filing › Standard deduction, pages 3–4\n"
    "The amounts are: $15,750.\n"
    "\n"
    "Question: What is the standard deduction?"
)


def helper_instance() -> AnswerCommand:
    database = Database("postgresql://u:p@h/d")
    sealer = SecretSealer(KEY)
    return AnswerCommand(
        LlmGateway(LlmConnectionStore(database, sealer), OpenAiChatBox(), ClaudeBox()),
        SearchStore(database, sealer, SectionPaths(database, sealer)),
        EmbeddingBox(EmbeddingSettings(model="BAAI/bge-small-en-v1.5", query_prefix="Q: ", batch_size=32)),
    )


def helper_user() -> UserAccount:
    return UserAccount(user_id=7, email="owner@example.com", display_name="Owner", is_disabled=False)


def helper_hits(best: float) -> list[SearchHit]:
    return [
        SearchHit(1, 40, 7, 3, "Publication 501", 12, "Standard Deduction", 25, 25, "Single $15,750", 0, best),
        SearchHit(
            2, 90, 31, 1, "Publication 17", 20, "Filing › Standard deduction", 3, 4, "The amounts are: $15,750.", 0, 0.7
        ),
    ]


def test___init__() -> None:
    database = Database("postgresql://u:p@h/d")
    sealer = SecretSealer(KEY)
    llm = LlmGateway(LlmConnectionStore(database, sealer), OpenAiChatBox(), ClaudeBox())
    store = SearchStore(database, sealer, SectionPaths(database, sealer))
    embeddings = EmbeddingBox(EmbeddingSettings(model="BAAI/bge-small-en-v1.5", query_prefix="Q: ", batch_size=32))
    tested = AnswerCommand(llm, store, embeddings)
    result = (tested._llm, tested._store, tested._embeddings)
    expected = (llm, store, embeddings)
    assert result == expected


def test_answered() -> None:
    tested = helper_instance()
    reply = '{"found": true, "answer": "$15,750.", "passages": [1]}'
    with (
        patch("rag_my_docs.commands.answer_command.time") as mock_time,
        patch.object(LlmGateway, "is_configured", side_effect=[True]) as mock_is_configured,
        patch.object(EmbeddingBox, "space", side_effect=[SPACE]) as mock_space,
        patch.object(SearchStore, "indexed_count", side_effect=[12]) as mock_indexed_count,
        patch.object(EmbeddingBox, "query", side_effect=[[1.0]]) as mock_query,
        patch.object(SearchStore, "nearest", side_effect=[helper_hits(0.78)]) as mock_nearest,
        patch.object(LlmGateway, "answer_reply", side_effect=[(reply, "good")]) as mock_answer_reply,
    ):
        mock_time.perf_counter.side_effect = [10.0, 10.5, 10.75, 14.75]
        result = tested.answered(helper_user(), "  What is the standard deduction?  ", 2)
    expected = AnswerResult(
        question="What is the standard deduction?",
        passages=tuple(helper_hits(0.78)),
        system=SYSTEM,
        prompt=PROMPT,
        answer=ModelAnswer(True, "$15,750.", (AnswerCitation(1, "Publication 501", 25),)),
        model="good",
        embedding_ms=500.0,
        search_ms=250.0,
        answer_ms=4000.0,
    )
    assert result == expected
    exp_calls = [call.perf_counter(), call.perf_counter(), call.perf_counter(), call.perf_counter()]
    assert mock_time.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_is_configured.mock_calls == exp_calls
    assert mock_space.mock_calls == exp_calls
    exp_calls = [call(7, SPACE)]
    assert mock_indexed_count.mock_calls == exp_calls
    exp_calls = [call("What is the standard deduction?")]
    assert mock_query.mock_calls == exp_calls
    exp_calls = [call(7, [1.0], SPACE, 2)]
    assert mock_nearest.mock_calls == exp_calls
    exp_calls = [call(SYSTEM, PROMPT)]
    assert mock_answer_reply.mock_calls == exp_calls


@pytest.mark.parametrize(
    "hits",
    [
        pytest.param(helper_hits(0.65), id="closest-below-the-threshold"),
        pytest.param([], id="no-passage"),
    ],
)
def test_answered__not_covered(hits: list[SearchHit]) -> None:
    """No passage close enough: the publications do not cover the question,
    and no model is called."""
    tested = helper_instance()
    with (
        patch("rag_my_docs.commands.answer_command.time") as mock_time,
        patch.object(LlmGateway, "is_configured", side_effect=[True]) as mock_is_configured,
        patch.object(EmbeddingBox, "space", side_effect=[SPACE]) as mock_space,
        patch.object(SearchStore, "indexed_count", side_effect=[12]) as mock_indexed_count,
        patch.object(EmbeddingBox, "query", side_effect=[[1.0]]) as mock_query,
        patch.object(SearchStore, "nearest", side_effect=[hits]) as mock_nearest,
        patch.object(LlmGateway, "answer_reply", side_effect=[]) as mock_answer_reply,
    ):
        mock_time.perf_counter.side_effect = [10.0, 10.5, 10.75]
        result = tested.answered(helper_user(), "How do I bake bread?", 5)
    expected = AnswerResult(
        question="How do I bake bread?",
        passages=tuple(hits),
        system="",
        prompt="",
        answer=ModelAnswer(False, "The publications do not cover this question: no passage is close enough to it.", ()),
        model="",
        embedding_ms=500.0,
        search_ms=250.0,
        answer_ms=0.0,
    )
    assert result == expected
    exp_calls = [call.perf_counter(), call.perf_counter(), call.perf_counter()]
    assert mock_time.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_is_configured.mock_calls == exp_calls
    assert mock_space.mock_calls == exp_calls
    exp_calls = [call(7, SPACE)]
    assert mock_indexed_count.mock_calls == exp_calls
    exp_calls = [call("How do I bake bread?")]
    assert mock_query.mock_calls == exp_calls
    exp_calls = [call(7, [1.0], SPACE, 5)]
    assert mock_nearest.mock_calls == exp_calls
    exp_calls = []
    assert mock_answer_reply.mock_calls == exp_calls


def test_answered__no_model_configured() -> None:
    tested = helper_instance()
    with (
        patch.object(LlmGateway, "is_configured", side_effect=[False]) as mock_is_configured,
        patch.object(SearchStore, "indexed_count", side_effect=[]) as mock_indexed_count,
        pytest.raises(AppException) as exc_info,
    ):
        tested.answered(helper_user(), "Who won?", 5)
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (503, "Answers need a language model: choose one in Settings › Language models.")
    assert result == expected
    exp_calls = [call()]
    assert mock_is_configured.mock_calls == exp_calls
    exp_calls = []
    assert mock_indexed_count.mock_calls == exp_calls


def test_answered__nothing_indexed() -> None:
    tested = helper_instance()
    with (
        patch.object(LlmGateway, "is_configured", side_effect=[True]) as mock_is_configured,
        patch.object(EmbeddingBox, "space", side_effect=[SPACE]) as mock_space,
        patch.object(SearchStore, "indexed_count", side_effect=[0]) as mock_indexed_count,
        patch.object(EmbeddingBox, "query", side_effect=[]) as mock_query,
        pytest.raises(AppException) as exc_info,
    ):
        tested.answered(helper_user(), "Who won?", 5)
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (409, "No publication is indexed yet: they are downloaded and indexed after the app starts.")
    assert result == expected
    exp_calls = [call()]
    assert mock_is_configured.mock_calls == exp_calls
    assert mock_space.mock_calls == exp_calls
    exp_calls = [call(7, SPACE)]
    assert mock_indexed_count.mock_calls == exp_calls
    exp_calls = []
    assert mock_query.mock_calls == exp_calls


@pytest.mark.parametrize(
    "question, top_k, expected",
    [
        pytest.param("  Who won?\n", 5, "Who won?", id="trimmed"),
        pytest.param("x" * 1000, 1, "x" * 1000, id="longest-and-fewest"),
        pytest.param("Who won?", 50, "Who won?", id="most-passages"),
    ],
)
def test_checked(question: str, top_k: int, expected: str) -> None:
    tested = AnswerCommand
    result = tested.checked(question, top_k)
    assert result == expected


@pytest.mark.parametrize(
    "question, top_k, expected",
    [
        pytest.param("   ", 5, (400, "Type a question to answer."), id="empty"),
        pytest.param("x" * 1001, 5, (400, "A question has at most 1000 characters."), id="too-long"),
        pytest.param("Who won?", 0, (400, "The number of passages is between 1 and 50."), id="no-passage"),
        pytest.param("Who won?", 51, (400, "The number of passages is between 1 and 50."), id="too-many"),
    ],
)
def test_checked__refused(question: str, top_k: int, expected: tuple[int, str]) -> None:
    tested = AnswerCommand
    with pytest.raises(AppException) as exc_info:
        tested.checked(question, top_k)
    result = (exc_info.value.status_code, exc_info.value.message)
    assert result == expected


def test_prompt() -> None:
    tested = AnswerCommand
    result = tested.prompt(helper_hits(0.78), "What is the standard deduction?")
    expected = PROMPT
    assert result == expected


@pytest.mark.parametrize(
    "start_page, end_page, expected",
    [
        pytest.param(25, 25, "page 25", id="one-page"),
        pytest.param(3, 4, "pages 3–4", id="across-pages"),
    ],
)
def test_pages(start_page: int, end_page: int, expected: str) -> None:
    tested = AnswerCommand
    result = tested.pages(SearchHit(1, 1, 0, 1, "a", 1, "s", start_page, end_page, "t", 0, 0.8))
    assert result == expected


@pytest.mark.parametrize(
    "reply, expected",
    [
        pytest.param(
            '{"found": true, "answer": " $15,750. ", "passages": [2, 1]}',
            ModelAnswer(
                True, "$15,750.", (AnswerCitation(2, "Publication 17", 3), AnswerCitation(1, "Publication 501", 25))
            ),
            id="cited",
        ),
        pytest.param(
            'Here it is: {"found": false, "answer": "Not in these passages."}',
            ModelAnswer(False, "Not in these passages.", ()),
            id="not-found-wrapped",
        ),
        pytest.param(" $15,750 ", ModelAnswer(True, "$15,750", ()), id="plain-text"),
        pytest.param('{"found": true}', ModelAnswer(True, '{"found": true}', ()), id="no-answer-field"),
    ],
)
def test_parsed_answer(reply: str, expected: ModelAnswer) -> None:
    tested = AnswerCommand
    result = tested.parsed_answer(reply, helper_hits(0.78))
    assert result == expected


@pytest.mark.parametrize(
    "numbers, expected",
    [
        pytest.param(
            [1, 2], (AnswerCitation(1, "Publication 501", 25), AnswerCitation(2, "Publication 17", 3)), id="both"
        ),
        pytest.param([2, 2, 9, "1", True], (AnswerCitation(2, "Publication 17", 3),), id="repeated-unknown-not-int"),
        pytest.param("1", (), id="not-a-list"),
        pytest.param(None, (), id="absent"),
    ],
)
def test_citations(numbers: Any, expected: tuple[AnswerCitation, ...]) -> None:
    tested = AnswerCommand
    result = tested.citations(numbers, helper_hits(0.78))
    assert result == expected


@pytest.mark.parametrize(
    "reply, expected",
    [
        pytest.param('{"a": 1}', {"a": 1}, id="bare"),
        pytest.param('Sure! {"a": {"b": 2}} Hope it helps.', {"a": {"b": 2}}, id="wrapped"),
    ],
)
def test_json_object(reply: str, expected: dict[str, object]) -> None:
    tested = AnswerCommand
    result = tested.json_object(reply)
    assert result == expected


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("no object here", id="no-braces"),
        pytest.param("} then {", id="reversed"),
        pytest.param("{not json}", id="invalid"),
    ],
)
def test_json_object__refused(reply: str) -> None:
    tested = AnswerCommand
    with pytest.raises(ValueError):
        tested.json_object(reply)
