from __future__ import annotations

import io
import json
import socket
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from rag_my_docs.libraries import openai_chat_box as openai_chat_box_module


class FakeOpenAiChat:
    """A hand-written stand-in for an OpenAI-compatible chat-completions
    server (a local vLLM, a hosted provider), reached the way `OpenAiChatBox`
    reaches it: `urllib.request.urlopen(request, timeout=...)`.

    It reproduces the exchange OpenAiChatBox depends on: a POST whose JSON body names
    the model and the messages, answered with `{"choices": [{"message":
    {"content": ...}}]}`. Its outcomes are given in order: a string is a
    reply's content, bytes are a raw body (a broken server), an exception is
    raised as urllib raises it (`HTTPError` for a status, `URLError` for an
    unreachable host, `TimeoutError`). Every request is recorded in `requests`:
    URL, method, the Authorization header, the decoded body and the timeout."""

    def __init__(self, outcomes: list[str | bytes | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[tuple[Any, ...]] = []

    @contextmanager
    def installed(self) -> Iterator[FakeOpenAiChat]:
        """In place of urlopen as openai_chat_box resolves it, for the block."""
        with patch.object(openai_chat_box_module.urllib.request, "urlopen", new=self):
            yield self

    def __call__(
        self,
        url: urllib.request.Request | str,
        data: bytes | None = None,
        timeout: float | object = socket._GLOBAL_DEFAULT_TIMEOUT,  # type: ignore[attr-defined]
        **kwargs: Any,
    ) -> io.BytesIO:
        request = url if isinstance(url, urllib.request.Request) else urllib.request.Request(url, data=data)
        body = request.data if isinstance(request.data, bytes) else b"{}"
        self.requests.append(
            (request.full_url, request.get_method(), request.get_header("Authorization"), json.loads(body), timeout)
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, bytes):
            return io.BytesIO(outcome)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": outcome}}]}).encode("utf-8"))
