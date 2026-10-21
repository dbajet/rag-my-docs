from __future__ import annotations

from unittest.mock import call, patch

import pytest

from rag_my_docs.constants.llm_kind import LlmKind
from rag_my_docs.libraries.claude_box import ClaudeBox
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.llm_connection_store import LlmConnectionStore
from rag_my_docs.libraries.llm_gateway import LlmGateway
from rag_my_docs.libraries.openai_chat_box import OpenAiChatBox
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.connection_health import ConnectionHealth
from rag_my_docs.structures.llm_connection import LlmConnection
from rag_my_docs.structures.llm_routing import LlmRouting

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="


def helper_instance() -> LlmGateway:
    return LlmGateway(
        LlmConnectionStore(Database("postgresql://u:p@h/d"), SecretSealer(KEY)), OpenAiChatBox(), ClaudeBox()
    )


def helper_connection(connection_id: int, kind: LlmKind) -> LlmConnection:
    return LlmConnection(connection_id, f"c{connection_id}", kind, "", "k", "good", True)


def test___init__() -> None:
    store = LlmConnectionStore(Database("postgresql://u:p@h/d"), SecretSealer(KEY))
    openai_box = OpenAiChatBox()
    claude_box = ClaudeBox()
    tested = LlmGateway(store, openai_box, claude_box)
    result = (tested._store, tested._openai_box, tested._claude_box)
    expected = (store, openai_box, claude_box)
    assert result == expected


@pytest.mark.parametrize(
    "routing, expected",
    [
        pytest.param(LlmRouting(2), True, id="chosen"),
        pytest.param(LlmRouting(None), False, id="none"),
    ],
)
def test_is_configured(routing: LlmRouting, expected: bool) -> None:
    tested = helper_instance()
    with patch.object(LlmConnectionStore, "routing", side_effect=[routing]) as mock_routing:
        result = tested.is_configured()
    assert result is expected
    exp_calls = [call()]
    assert mock_routing.mock_calls == exp_calls


def test_answer_reply__claude() -> None:
    tested = helper_instance()
    with (
        patch.object(LlmConnectionStore, "routing", side_effect=[LlmRouting(2)]) as mock_routing,
        patch.object(LlmGateway, "connection_for", side_effect=[helper_connection(2, LlmKind.CLAUDE)]) as mock_for,
        patch.object(ClaudeBox, "reply", side_effect=["Ada."]) as mock_reply,
    ):
        result = tested.answer_reply("System.", "Passages.")
    expected = ("Ada.", "good")
    assert result == expected
    exp_calls = [call()]
    assert mock_routing.mock_calls == exp_calls
    exp_calls = [call(2)]
    assert mock_for.mock_calls == exp_calls
    exp_calls = [call(helper_connection(2, LlmKind.CLAUDE), "good", "System.", "Passages.", 16000, "medium", 120.0)]
    assert mock_reply.mock_calls == exp_calls


def test_answer_reply__openai() -> None:
    tested = helper_instance()
    with (
        patch.object(LlmConnectionStore, "routing", side_effect=[LlmRouting(2)]) as mock_routing,
        patch.object(LlmGateway, "connection_for", side_effect=[helper_connection(2, LlmKind.OPENAI)]) as mock_for,
        patch.object(OpenAiChatBox, "reply", side_effect=["Ada."]) as mock_reply,
    ):
        result = tested.answer_reply("System.", "Passages.")
    expected = ("Ada.", "good")
    assert result == expected
    exp_calls = [call()]
    assert mock_routing.mock_calls == exp_calls
    exp_calls = [call(2)]
    assert mock_for.mock_calls == exp_calls
    exp_calls = [call(helper_connection(2, LlmKind.OPENAI), "good", "System.", "Passages.", 1500, 120.0)]
    assert mock_reply.mock_calls == exp_calls


@pytest.mark.parametrize(
    "kind, box",
    [
        pytest.param(LlmKind.CLAUDE, ClaudeBox, id="claude"),
        pytest.param(LlmKind.OPENAI, OpenAiChatBox, id="openai"),
    ],
)
def test_health(kind: LlmKind, box: type[ClaudeBox] | type[OpenAiChatBox]) -> None:
    tested = helper_instance()
    light = ConnectionHealth(1, True, "Connected.", 10.0)
    with patch.object(box, "health", side_effect=[light]) as mock_health:
        result = tested.health(helper_connection(1, kind))
    expected = ConnectionHealth(1, True, "Connected.", 10.0)
    assert result == expected
    exp_calls = [call(helper_connection(1, kind), 8.0)]
    assert mock_health.mock_calls == exp_calls


@pytest.mark.parametrize(
    "kind, box",
    [
        pytest.param(LlmKind.CLAUDE, ClaudeBox, id="claude"),
        pytest.param(LlmKind.OPENAI, OpenAiChatBox, id="openai"),
    ],
)
def test_models(kind: LlmKind, box: type[ClaudeBox] | type[OpenAiChatBox]) -> None:
    tested = helper_instance()
    with patch.object(box, "models", side_effect=[["fast", "good"]]) as mock_models:
        result = tested.models(helper_connection(1, kind))
    expected = ["fast", "good"]
    assert result == expected
    exp_calls = [call(helper_connection(1, kind), 8.0)]
    assert mock_models.mock_calls == exp_calls


def test_connection_for() -> None:
    tested = helper_instance()
    with patch.object(
        LlmConnectionStore, "connection", side_effect=[helper_connection(4, LlmKind.OPENAI)]
    ) as mock_connection:
        result = tested.connection_for(4)
    expected = helper_connection(4, LlmKind.OPENAI)
    assert result == expected
    exp_calls = [call(4)]
    assert mock_connection.mock_calls == exp_calls


@pytest.mark.parametrize(
    "connection_id, found, exp_calls",
    [
        pytest.param(None, [], [], id="no-step"),
        pytest.param(9, [None], [call(9)], id="deleted-meanwhile"),
    ],
)
def test_connection_for__none(connection_id: int | None, found: list[None], exp_calls: list[object]) -> None:
    tested = helper_instance()
    with (
        patch.object(LlmConnectionStore, "connection", side_effect=found) as mock_connection,
        pytest.raises(AppException) as exc_info,
    ):
        tested.connection_for(connection_id)
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (503, "Answers need a language model: choose one in Settings › Language models.")
    assert result == expected
    assert mock_connection.mock_calls == exp_calls
