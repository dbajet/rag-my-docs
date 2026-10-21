from __future__ import annotations

import logging
from typing import Any
from unittest.mock import call, patch

import anthropic
import pytest

from rag_my_docs.constants.llm_kind import LlmKind
from rag_my_docs.libraries.claude_box import ClaudeBox
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.connection_health import ConnectionHealth
from rag_my_docs.structures.llm_connection import LlmConnection
from tests.support.fake_anthropic import FakeAnthropic


def helper_connection(model: str) -> LlmConnection:
    return LlmConnection(
        connection_id=4,
        name="Claude",
        kind=LlmKind.CLAUDE,
        base_url="",
        api_key="sk-ant-key",
        model=model,
        disable_thinking=True,
    )


def helper_request(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 16000,
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "Hi?"}],
        "output_config": {"effort": "medium"},
    }


def test_reply() -> None:
    """A model with a server-side fallback gets it; thinking blocks are left out."""
    tested = ClaudeBox()
    fake = FakeAnthropic([FakeAnthropic.message("end_turn", ["Ada ", "won."])])
    with fake.installed():
        result = tested.reply(
            helper_connection("claude-opus-5-5"),
            "claude-opus-5-5",
            "Be brief.",
            "Hi?",
            16000,
            "medium",
            50.0,
        )
    expected = "Ada won."
    assert result == expected
    exp_calls = [
        ("client", "sk-ant-key", 50.0, 0),
        (
            "messages.create",
            helper_request("claude-opus-5-5") | {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"},
        ),
    ]
    assert fake.calls == exp_calls


def test_reply__a_model_without_fallback() -> None:
    tested = ClaudeBox()
    fake = FakeAnthropic([FakeAnthropic.message("end_turn", ["{}"])])
    with fake.installed():
        result = tested.reply(
            helper_connection("claude-haiku-5-5"),
            "claude-haiku-5-5",
            "Be brief.",
            "Hi?",
            16000,
            "medium",
            30.0,
        )
    expected = "{}"
    assert result == expected
    exp_calls = [("client", "sk-ant-key", 30.0, 0), ("messages.create", helper_request("claude-haiku-5-5"))]
    assert fake.calls == exp_calls


def test_reply__a_refusal_is_not_an_answer() -> None:
    tested = ClaudeBox()
    fake = FakeAnthropic([FakeAnthropic.message("refusal", [])])
    with fake.installed(), pytest.raises(AppException) as exc_info:
        tested.reply(helper_connection("m"), "claude-haiku-5-5", "Be brief.", "Hi?", 16000, "medium", 30.0)
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (422, "The language model declined to answer this question.")
    assert result == expected
    exp_calls = [("client", "sk-ant-key", 30.0, 0), ("messages.create", helper_request("claude-haiku-5-5"))]
    assert fake.calls == exp_calls


@pytest.mark.parametrize(
    "error, exp_kind",
    [
        pytest.param(FakeAnthropic.status_error(anthropic.AuthenticationError, 401), "AuthenticationError", id="key"),
        pytest.param(FakeAnthropic.status_error(anthropic.RateLimitError, 429), "RateLimitError", id="rate-limit"),
        pytest.param(
            FakeAnthropic.status_error(anthropic.InternalServerError, 500), "InternalServerError", id="server"
        ),
        pytest.param(
            FakeAnthropic.connection_error(anthropic.APIConnectionError), "APIConnectionError", id="unreachable"
        ),
        pytest.param(FakeAnthropic.connection_error(anthropic.APITimeoutError), "APITimeoutError", id="timeout"),
    ],
)
def test_reply__a_failure_is_unavailable(caplog: pytest.LogCaptureFixture, error: BaseException, exp_kind: str) -> None:
    tested = ClaudeBox()
    fake = FakeAnthropic([error])
    with fake.installed(), caplog.at_level(logging.WARNING, logger="rag_my_docs"):
        with pytest.raises(AppException) as exc_info:
            tested.reply(helper_connection("m"), "claude-haiku-5-5", "Be brief.", "Hi?", 16000, "medium", 30.0)
    result = (exc_info.value.status_code, exc_info.value.message, exc_info.value.__cause__)
    expected = (503, "The language model did not answer. Try again in a moment.", None)
    assert result == expected
    exp_messages = [f"[LLM] {exp_kind} from claude-haiku-5-5"]
    assert [record.getMessage() for record in caplog.records] == exp_messages


@pytest.mark.parametrize(
    "model, outcomes, expected, exp_retrieved",
    [
        pytest.param(
            "claude-opus-5-5",
            [FakeAnthropic.model("claude-opus-5-5")],
            (True, "Connected: the key is accepted and the model is there."),
            ["claude-opus-5-5"],
            id="model-found",
        ),
        pytest.param(
            "claude-nope",
            [FakeAnthropic.status_error(anthropic.NotFoundError, 404)],
            (False, "Claude has no model claude-nope."),
            ["claude-nope"],
            id="unknown-model",
        ),
        pytest.param(
            "claude-haiku-5-5",
            [FakeAnthropic.status_error(anthropic.AuthenticationError, 401)],
            (False, "The API key was refused."),
            ["claude-haiku-5-5"],
            id="key-refused",
        ),
        pytest.param(
            "claude-haiku-5-5",
            [FakeAnthropic.status_error(anthropic.PermissionDeniedError, 403)],
            (False, "The API key was refused."),
            ["claude-haiku-5-5"],
            id="key-not-allowed",
        ),
        pytest.param(
            "claude-haiku-5-5",
            [FakeAnthropic.connection_error(anthropic.APITimeoutError)],
            (False, "Unreachable: no answer from the address."),
            ["claude-haiku-5-5"],
            id="unreachable",
        ),
        pytest.param(
            "claude-haiku-5-5",
            [FakeAnthropic.status_error(anthropic.InternalServerError, 503)],
            (False, "Claude answered HTTP 503."),
            ["claude-haiku-5-5"],
            id="server-error",
        ),
    ],
)
def test_health(model: str, outcomes: list[Any], expected: tuple[bool, str], exp_retrieved: list[str]) -> None:
    tested = ClaudeBox()
    fake = FakeAnthropic(outcomes)
    with fake.installed(), patch("rag_my_docs.libraries.claude_box.time") as mock_time:
        mock_time.perf_counter.side_effect = [10.0, 10.25]
        health = tested.health(helper_connection(model), 8.0)
    result = (health.is_ok, health.detail)
    assert result == expected
    exp_health = ConnectionHealth(4, expected[0], expected[1], 250.0)
    assert health == exp_health
    exp_calls = [("client", "sk-ant-key", 8.0, 0), *[("models.retrieve", model) for model in exp_retrieved]]
    assert fake.calls == exp_calls
    exp_time = [call.perf_counter(), call.perf_counter()]
    assert mock_time.mock_calls == exp_time


def test_models() -> None:
    tested = ClaudeBox()
    fake = FakeAnthropic([FakeAnthropic.model_page(["claude-opus-5-5", "claude-haiku-5-5"])])
    with fake.installed():
        result = tested.models(helper_connection(""), 8.0)
    expected = ["claude-opus-5-5", "claude-haiku-5-5"]
    assert result == expected
    exp_calls = [("client", "sk-ant-key", 8.0, 0), ("models.list", {"limit": 1000})]
    assert fake.calls == exp_calls


@pytest.mark.parametrize(
    "error, exp_message",
    [
        pytest.param(
            FakeAnthropic.status_error(anthropic.AuthenticationError, 401), "The API key was refused.", id="key-refused"
        ),
        pytest.param(
            FakeAnthropic.status_error(anthropic.PermissionDeniedError, 403),
            "The API key was refused.",
            id="key-not-allowed",
        ),
        pytest.param(
            FakeAnthropic.connection_error(anthropic.APITimeoutError),
            "Unreachable: no answer from the address.",
            id="unreachable",
        ),
        pytest.param(
            FakeAnthropic.status_error(anthropic.InternalServerError, 503),
            "Claude answered HTTP 503.",
            id="server-error",
        ),
    ],
)
def test_models__refused(error: BaseException, exp_message: str) -> None:
    tested = ClaudeBox()
    fake = FakeAnthropic([error])
    with fake.installed(), pytest.raises(AppException) as exc_info:
        tested.models(helper_connection(""), 8.0)
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (502, exp_message)
    assert result == expected
    exp_calls = [("client", "sk-ant-key", 8.0, 0), ("models.list", {"limit": 1000})]
    assert fake.calls == exp_calls
