from __future__ import annotations

import logging
import urllib.error
from typing import Any
from unittest.mock import call, patch

import pytest

from rag_my_docs.constants.llm_kind import LlmKind
from rag_my_docs.libraries.openai_chat_box import OpenAiChatBox
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.connection_health import ConnectionHealth
from rag_my_docs.structures.llm_connection import LlmConnection
from tests.support.fake_openai_chat import FakeOpenAiChat

URL = "https://llm.example.com/v1/chat/completions"
MODELS = "https://llm.example.com/v1/models"


def helper_connection(api_key: str, disable_thinking: bool) -> LlmConnection:
    return LlmConnection(
        connection_id=3,
        name="Local",
        kind=LlmKind.OPENAI,
        base_url="https://llm.example.com/v1",
        api_key=api_key,
        model="good",
        disable_thinking=disable_thinking,
    )


def helper_body(max_tokens: int) -> dict[str, Any]:
    return {
        "model": "good",
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi?"}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_reply() -> None:
    tested = OpenAiChatBox()
    fake = FakeOpenAiChat(["Bonjour", b'{"choices": [{"message": {"content": null}}]}'])
    with fake.installed():
        result = (
            tested.reply(helper_connection("the-key", True), "good", "Be brief.", "Hi?", 1500, 50.0),
            tested.reply(helper_connection("", True), "good", "Be brief.", "Hi?", 10, 2.0),
        )
    # A null content (a reply cut before any text) reads as empty.
    expected = ("Bonjour", "")
    assert result == expected
    exp_requests = [
        (URL, "POST", "Bearer the-key", helper_body(1500), 50.0),
        # A local endpoint without a key gets no Authorization header.
        (URL, "POST", None, helper_body(10), 2.0),
    ]
    assert fake.requests == exp_requests


@pytest.mark.parametrize(
    "outcome, exp_kind",
    [
        pytest.param(
            urllib.error.HTTPError(URL, 500, "Server Error", None, None),  # type: ignore[arg-type]
            "HTTPError",
            id="server-error",
        ),
        pytest.param(urllib.error.URLError("no route to host"), "URLError", id="unreachable"),
        pytest.param(TimeoutError("timed out"), "TimeoutError", id="timeout"),
        pytest.param(b"<html>Bad gateway</html>", "JSONDecodeError", id="not-json"),
        pytest.param(b'{"error": "overloaded"}', "KeyError", id="no-choices"),
        pytest.param(b'{"choices": []}', "IndexError", id="empty-choices"),
    ],
)
def test_reply__a_failure_is_unavailable(
    caplog: pytest.LogCaptureFixture,
    outcome: str | bytes | BaseException,
    exp_kind: str,
) -> None:
    """The person reads that the model did not answer; the log says how,
    never with the provider's text."""
    tested = OpenAiChatBox()
    fake = FakeOpenAiChat([outcome])
    with fake.installed(), caplog.at_level(logging.WARNING, logger="rag_my_docs"):
        with pytest.raises(AppException) as exc_info:
            tested.reply(helper_connection("the-key", True), "good", "Be brief.", "Hi?", 10, 2.0)
    result = (exc_info.value.status_code, exc_info.value.message, exc_info.value.__cause__)
    expected = (503, "The language model did not answer. Try again in a moment.", None)
    assert result == expected
    exp_messages = [f"[LLM] {exp_kind} from good"]
    assert [record.getMessage() for record in caplog.records] == exp_messages


@pytest.mark.parametrize(
    "outcome, expected",
    [
        pytest.param(
            b'{"data": [{"id": "fast"}, {"id": "good"}, {"id": "other"}]}',
            (True, "Connected: the key is accepted and the model is there."),
            id="model-served",
        ),
        pytest.param(b'{"data": [{"id": "fast"}]}', (False, "The endpoint does not serve good."), id="model-missing"),
        pytest.param(
            urllib.error.HTTPError(MODELS, 401, "Unauthorized", None, None),  # type: ignore[arg-type]
            (False, "The API key was refused."),
            id="key-refused",
        ),
        pytest.param(
            urllib.error.HTTPError(MODELS, 502, "Bad Gateway", None, None),  # type: ignore[arg-type]
            (False, "The endpoint answered HTTP 502."),
            id="server-error",
        ),
        pytest.param(
            urllib.error.URLError("timed out"), (False, "Unreachable: no answer from the address."), id="unreachable"
        ),
        pytest.param(TimeoutError("timed out"), (False, "Unreachable: no answer from the address."), id="timeout"),
        pytest.param(
            b"<html>Welcome</html>",
            (False, "The endpoint's answer is not an OpenAI-compatible model list."),
            id="not-a-model-list",
        ),
    ],
)
def test_health(outcome: str | bytes | BaseException, expected: tuple[bool, str]) -> None:
    tested = OpenAiChatBox()
    fake = FakeOpenAiChat([outcome])
    with fake.installed(), patch("rag_my_docs.libraries.openai_chat_box.time") as mock_time:
        mock_time.perf_counter.side_effect = [3.0, 3.125]
        health = tested.health(helper_connection("the-key", True), 8.0)
    result = (health.is_ok, health.detail)
    assert result == expected
    exp_health = ConnectionHealth(3, expected[0], expected[1], 125.0)
    assert health == exp_health
    exp_requests = [(MODELS, "GET", "Bearer the-key", {}, 8.0)]
    assert fake.requests == exp_requests
    exp_time = [call.perf_counter(), call.perf_counter()]
    assert mock_time.mock_calls == exp_time


def test_models() -> None:
    tested = OpenAiChatBox()
    fake = FakeOpenAiChat([b'{"data": [{"id": "good"}, {"id": "fast"}, {"id": "good"}]}'])
    with fake.installed():
        result = tested.models(helper_connection("", True), 8.0)
    expected = ["fast", "good"]
    assert result == expected
    exp_requests = [(MODELS, "GET", None, {}, 8.0)]
    assert fake.requests == exp_requests


@pytest.mark.parametrize(
    "outcome, exp_message",
    [
        pytest.param(
            urllib.error.HTTPError(MODELS, 403, "Forbidden", None, None),  # type: ignore[arg-type]
            "The API key was refused.",
            id="key-refused",
        ),
        pytest.param(
            urllib.error.HTTPError(MODELS, 404, "Not Found", None, None),  # type: ignore[arg-type]
            "The endpoint answered HTTP 404.",
            id="wrong-address",
        ),
        pytest.param(urllib.error.URLError("refused"), "Unreachable: no answer from the address.", id="unreachable"),
        pytest.param(
            b'{"models": []}', "The endpoint's answer is not an OpenAI-compatible model list.", id="not-a-model-list"
        ),
    ],
)
def test_models__refused(outcome: bytes | BaseException, exp_message: str) -> None:
    tested = OpenAiChatBox()
    fake = FakeOpenAiChat([outcome])
    with fake.installed(), pytest.raises(AppException) as exc_info:
        tested.models(helper_connection("the-key", True), 8.0)
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (502, exp_message)
    assert result == expected
    exp_requests = [(MODELS, "GET", "Bearer the-key", {}, 8.0)]
    assert fake.requests == exp_requests


@pytest.mark.parametrize(
    "api_key, expected",
    [
        pytest.param("k", {"Authorization": "Bearer k"}, id="key"),
        pytest.param("", {}, id="no-key"),
    ],
)
def test_headers(api_key: str, expected: dict[str, str]) -> None:
    tested = OpenAiChatBox
    result = tested.headers(helper_connection(api_key, True))
    assert result == expected


@pytest.mark.parametrize(
    "disable_thinking, expected",
    [
        pytest.param(True, helper_body(10), id="thinking-off"),
        pytest.param(
            False,
            {
                "model": "good",
                "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi?"}],
                "max_tokens": 10,
                "temperature": 0,
            },
            id="provider-default",
        ),
    ],
)
def test_body(disable_thinking: bool, expected: dict[str, object]) -> None:
    tested = OpenAiChatBox
    result = tested.body(helper_connection("k", disable_thinking), "good", "Be brief.", "Hi?", 10)
    assert result == expected
