from __future__ import annotations

from unittest.mock import call, patch

from starlette.requests import Request

from rag_my_docs.commands.answer_command import AnswerCommand
from rag_my_docs.commands.evaluation_command import EvaluationCommand
from rag_my_docs.commands.session_command import SessionCommand
from rag_my_docs.handlers.evaluation_router import EvaluationRouter
from rag_my_docs.handlers.evaluation_run_request import EvaluationRunRequest
from rag_my_docs.handlers.session_guard import SessionGuard
from rag_my_docs.libraries.blind_index import BlindIndex
from rag_my_docs.libraries.claude_box import ClaudeBox
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.embedding_box import EmbeddingBox
from rag_my_docs.libraries.llm_connection_store import LlmConnectionStore
from rag_my_docs.libraries.llm_gateway import LlmGateway
from rag_my_docs.libraries.openai_chat_box import OpenAiChatBox
from rag_my_docs.libraries.search_store import SearchStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.libraries.session_store import SessionStore
from rag_my_docs.libraries.user_store import UserStore
from rag_my_docs.structures.answer_result import AnswerResult
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.evaluation_case import EvaluationCase
from rag_my_docs.structures.evaluation_outcome import EvaluationOutcome
from rag_my_docs.structures.model_answer import ModelAnswer
from rag_my_docs.structures.settings import Settings
from rag_my_docs.structures.user_account import UserAccount

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="


def helper_settings() -> Settings:
    return Settings(
        environment="dev",
        database_url="postgresql://u:p@h/d",
        app_version="v",
        build_time="b",
        app_commit="c",
        app_base_url="https://docs.example.com",
        data_key=KEY,
        smtp_host="",
        smtp_port=587,
        smtp_user="",
        smtp_password="",
        email_from="",
        bootstrap_admin_email="",
        chunking=ChunkingSettings(target_tokens=200, max_tokens=350, min_tokens=75, overlap_sentences=1),
        embedding=EmbeddingSettings(model="BAAI/bge-small-en-v1.5", query_prefix="Q: ", batch_size=32),
    )


def helper_parts() -> tuple[SessionGuard, EvaluationCommand]:
    settings = helper_settings()
    database = Database(settings.database_url)
    sealer = SecretSealer(KEY)
    guard = SessionGuard(settings, SessionCommand(UserStore(database, sealer, BlindIndex(KEY)), SessionStore(database)))
    answers = AnswerCommand(
        LlmGateway(LlmConnectionStore(database, sealer), OpenAiChatBox(), ClaudeBox()),
        SearchStore(database, sealer, SectionPaths(database, sealer)),
        EmbeddingBox(settings.embedding),
    )
    return guard, EvaluationCommand(answers, [EvaluationCase("Who won?", ("2016.pdf",), ("Ada",), True)])


def helper_user() -> UserAccount:
    return UserAccount(user_id=1, email="owner@example.com", display_name="Owner", is_disabled=False)


def helper_request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/api/evaluation/run", "headers": []})


def test___init__() -> None:
    guard, evaluations = helper_parts()
    tested = EvaluationRouter(guard, evaluations)
    result = (tested._guard, tested._evaluations)
    expected = (guard, evaluations)
    assert result == expected


def test_router() -> None:
    tested = EvaluationRouter(*helper_parts())
    result = [(route.path, tuple(sorted(route.methods)), route.name) for route in tested.router().routes]  # type: ignore[attr-defined]
    expected = [
        ("/api/evaluation/cases", ("GET",), "cases"),
        ("/api/evaluation/run", ("POST",), "run"),
    ]
    assert result == expected


def test_cases() -> None:
    tested = EvaluationRouter(*helper_parts())
    request = helper_request()
    with (
        patch.object(SessionGuard, "user", side_effect=[helper_user()]) as mock_user,
        patch.object(
            EvaluationCommand, "cases", side_effect=[[EvaluationCase("Who won?", ("2016.pdf",), ("Ada",), True)]]
        ) as mock_cases,
    ):
        result = tested.cases(request)
    expected = {
        "cases": [{"question": "Who won?", "documents": ["2016.pdf"], "facts": ["Ada"], "expects_answer": True}]
    }
    assert result == expected
    exp_calls = [call(request)]
    assert mock_user.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_cases.mock_calls == exp_calls


def test_run() -> None:
    tested = EvaluationRouter(*helper_parts())
    request = helper_request()
    outcome = EvaluationOutcome(
        "Who won?",
        (),
        (),
        True,
        True,
        AnswerResult("Who won?", (), "rules", "prompt", ModelAnswer(True, "Ada.", ()), "good", 1.0, 2.0, 3.0),
    )
    with (
        patch.object(SessionGuard, "user", side_effect=[helper_user()]) as mock_user,
        patch.object(EvaluationCommand, "evaluated", side_effect=[outcome]) as mock_evaluated,
    ):
        result = tested.run(request, EvaluationRunRequest(index=0, top_k=10))
    expected = {
        "question": "Who won?",
        "missing_documents": [],
        "missing_facts": [],
        "is_retrieval_right": True,
        "is_answer_right": True,
        "result": {
            "question": "Who won?",
            "passages": [],
            "system": "rules",
            "prompt": "prompt",
            "answer": {"is_found": True, "text": "Ada.", "citations": []},
            "model": "good",
            "embedding_ms": 1.0,
            "search_ms": 2.0,
            "answer_ms": 3.0,
        },
    }
    assert result == expected
    exp_calls = [call(request)]
    assert mock_user.mock_calls == exp_calls
    exp_calls = [call(helper_user(), 0, 10)]
    assert mock_evaluated.mock_calls == exp_calls
