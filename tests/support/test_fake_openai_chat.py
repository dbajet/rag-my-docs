from __future__ import annotations

import io
import socket
import urllib.error
import urllib.request

import pytest

from rag_my_docs.libraries import openai_chat_box as openai_chat_box_module
from tests.support.fake_openai_chat import FakeOpenAiChat


def helper_request() -> urllib.request.Request:
    return urllib.request.Request(
        "https://llm.example.com/v1/chat/completions",
        data=b'{"model": "m"}',
        headers={"Authorization": "Bearer k"},
        method="POST",
    )


def test___init__() -> None:
    tested = FakeOpenAiChat(["a", b"b"])
    result = (tested.outcomes, tested.requests)
    expected = (["a", b"b"], [])
    assert result == expected


def test_installed() -> None:
    tested = FakeOpenAiChat([])
    original = openai_chat_box_module.urllib.request.urlopen
    with tested.installed() as installed:
        result = (installed is tested, openai_chat_box_module.urllib.request.urlopen is tested)
    expected = (True, True)
    assert result == expected
    exp_restored = original
    assert openai_chat_box_module.urllib.request.urlopen is exp_restored


def test___call__() -> None:
    tested = FakeOpenAiChat(["Hello", b"not json"])
    result = (tested(helper_request(), timeout=5.0).read(), tested(helper_request(), timeout=6.0).read())
    expected = (b'{"choices": [{"message": {"content": "Hello"}}]}', b"not json")
    assert result == expected
    exp_requests = [
        ("https://llm.example.com/v1/chat/completions", "POST", "Bearer k", {"model": "m"}, 5.0),
        ("https://llm.example.com/v1/chat/completions", "POST", "Bearer k", {"model": "m"}, 6.0),
    ]
    assert tested.requests == exp_requests
    exp_left: list[str | bytes | BaseException] = []
    assert tested.outcomes == exp_left


def test___call____a_url_and_no_body() -> None:
    tested = FakeOpenAiChat(["ok"])
    result = isinstance(tested("https://llm.example.com/v1/models"), io.BytesIO)
    assert result is True
    # No timeout given: urllib's own default sentinel.
    expected = [("https://llm.example.com/v1/models", "GET", None, {}, socket._GLOBAL_DEFAULT_TIMEOUT)]  # type: ignore[attr-defined]
    assert tested.requests == expected


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            urllib.error.HTTPError("https://llm.example.com", 500, "Server Error", None, None),  # type: ignore[arg-type]
            id="status",
        ),
        pytest.param(urllib.error.URLError("unreachable"), id="unreachable"),
        pytest.param(TimeoutError("timed out"), id="timeout"),
    ],
)
def test___call____fails_as_urllib_does(error: BaseException) -> None:
    tested = FakeOpenAiChat([error])
    with pytest.raises(type(error)) as exc_info:
        tested(helper_request(), timeout=5.0)
    assert exc_info.value is error
    expected = [("https://llm.example.com/v1/chat/completions", "POST", "Bearer k", {"model": "m"}, 5.0)]
    assert tested.requests == expected
