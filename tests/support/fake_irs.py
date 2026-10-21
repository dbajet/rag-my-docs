from __future__ import annotations

import io
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from email.message import Message
from typing import Any
from unittest.mock import patch

from rag_my_docs.libraries import publication_box as publication_box_module


class FakeIrs:
    """A hand-written stand-in for the IRS file server, reached the way
    `PublicationBox` reaches it: `urlopen(request, timeout=...)`, imported by
    name into publication_box.

    It reproduces what the box depends on: a GET by URL answered with the
    file's bytes, read with a size limit, and a URL the server does not have
    answered with HTTP 404 (`urllib.error.HTTPError`, as urllib raises it).
    A file given as an exception is raised instead (`URLError`, `TimeoutError`).
    Every request is recorded in `requests`: URL, User-Agent and timeout."""

    def __init__(self, files: dict[str, bytes | BaseException]) -> None:
        self.files = dict(files)
        self.requests: list[tuple[Any, ...]] = []

    @contextmanager
    def installed(self) -> Iterator[FakeIrs]:
        """In place of urlopen as publication_box resolves it, for the block."""
        with patch.object(publication_box_module, "urlopen", new=self):
            yield self

    def __call__(
        self,
        url: urllib.request.Request | str,
        data: bytes | None = None,
        timeout: float | object = socket._GLOBAL_DEFAULT_TIMEOUT,  # type: ignore[attr-defined]
        **kwargs: Any,
    ) -> io.BytesIO:
        request = url if isinstance(url, urllib.request.Request) else urllib.request.Request(url)
        self.requests.append((request.full_url, request.get_header("User-agent"), timeout))
        found = self.files.get(request.full_url)
        if found is None:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", Message(), None)
        if isinstance(found, BaseException):
            raise found
        return io.BytesIO(found)
