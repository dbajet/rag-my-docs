from __future__ import annotations

from types import SimpleNamespace

import anthropic
import pytest

from rag_my_docs.libraries import claude_box as claude_box_module
from tests.support.fake_anthropic import FakeAnthropic


def test___init__() -> None:
    tested = FakeAnthropic(["a"])
    result = (tested.outcomes, tested.calls, tested.beta is tested, tested.messages is tested, tested.models is tested)
    expected = (["a"], [], True, True, True)
    assert result == expected


def test_installed() -> None:
    tested = FakeAnthropic([])
    original = claude_box_module.anthropic.Anthropic
    with tested.installed() as installed:
        result = (installed is tested, claude_box_module.anthropic.Anthropic is tested)
    expected = (True, True)
    assert result == expected
    exp_restored = original
    assert claude_box_module.anthropic.Anthropic is exp_restored


def test___call__() -> None:
    tested = FakeAnthropic([])
    result = tested(api_key="k", timeout=5.0, max_retries=0)
    assert result is tested
    expected = [("client", "k", 5.0, 0)]
    assert tested.calls == expected


def test_create() -> None:
    tested = FakeAnthropic([FakeAnthropic.message("end_turn", ["Hi"])])
    result = tested.beta.messages.create(model="m", max_tokens=10)
    expected = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="Hi")],
    )
    assert result == expected
    exp_calls = [("messages.create", {"model": "m", "max_tokens": 10})]
    assert tested.calls == exp_calls


def test_retrieve() -> None:
    error = FakeAnthropic.status_error(anthropic.NotFoundError, 404)
    tested = FakeAnthropic([FakeAnthropic.model("claude-opus-5-5"), error])
    result = tested.models.retrieve("claude-opus-5-5")
    expected = SimpleNamespace(id="claude-opus-5-5", type="model")
    assert result == expected
    with pytest.raises(anthropic.NotFoundError) as exc_info:
        tested.models.retrieve("claude-nope")
    assert exc_info.value is error
    exp_calls = [("models.retrieve", "claude-opus-5-5"), ("models.retrieve", "claude-nope")]
    assert tested.calls == exp_calls


def test_list() -> None:
    tested = FakeAnthropic([FakeAnthropic.model_page(["claude-opus-5-5"])])
    result = tested.models.list(limit=1000)
    expected = SimpleNamespace(data=[SimpleNamespace(id="claude-opus-5-5", type="model")], has_more=False)
    assert result == expected
    exp_calls = [("models.list", {"limit": 1000})]
    assert tested.calls == exp_calls


def test_next() -> None:
    tested = FakeAnthropic(["first", "second"])
    result = (tested.next(), tested.next())
    expected = ("first", "second")
    assert result == expected


def test_message() -> None:
    tested = FakeAnthropic
    result = tested.message("refusal", [])
    expected = SimpleNamespace(stop_reason="refusal", content=[SimpleNamespace(type="thinking", thinking="")])
    assert result == expected


def test_model() -> None:
    tested = FakeAnthropic
    result = tested.model("claude-haiku-5-5")
    expected = SimpleNamespace(id="claude-haiku-5-5", type="model")
    assert result == expected


def test_model_page() -> None:
    tested = FakeAnthropic
    result = tested.model_page(["claude-opus-5-5", "claude-haiku-5-5"])
    expected = SimpleNamespace(
        data=[
            SimpleNamespace(id="claude-opus-5-5", type="model"),
            SimpleNamespace(id="claude-haiku-5-5", type="model"),
        ],
        has_more=False,
    )
    assert result == expected


@pytest.mark.parametrize(
    "error_type, status",
    [
        pytest.param(anthropic.AuthenticationError, 401, id="authentication"),
        pytest.param(anthropic.NotFoundError, 404, id="not-found"),
        pytest.param(anthropic.InternalServerError, 500, id="server"),
    ],
)
def test_status_error(error_type: type[anthropic.APIStatusError], status: int) -> None:
    tested = FakeAnthropic
    error = tested.status_error(error_type, status)
    result = (type(error), error.status_code, error.message)
    expected = (error_type, status, f"Error code: {status}")
    assert result == expected


@pytest.mark.parametrize(
    "error_type, exp_message",
    [
        pytest.param(anthropic.APIConnectionError, "Connection error.", id="connection"),
        pytest.param(
            anthropic.APITimeoutError,
            "Request timed out or interrupted. This could be due to a network timeout, dropped connection, or "
            "request cancellation. See https://docs.anthropic.com/en/api/errors#long-requests for more details.",
            id="timeout",
        ),
    ],
)
def test_connection_error(error_type: type[anthropic.APIConnectionError], exp_message: str) -> None:
    tested = FakeAnthropic
    error = tested.connection_error(error_type)
    result = (type(error), error.message)
    expected = (error_type, exp_message)
    assert result == expected
