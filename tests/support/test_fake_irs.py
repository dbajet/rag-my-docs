from __future__ import annotations

import socket
import urllib.error
import urllib.request

import pytest

from rag_my_docs.libraries import publication_box as publication_box_module
from tests.support.fake_irs import FakeIrs

URL = "https://www.irs.gov/pub/irs-pdf/p17.pdf"


def test___init__() -> None:
    tested = FakeIrs({URL: b"pdf"})
    result = (tested.files, tested.requests)
    expected = ({URL: b"pdf"}, [])
    assert result == expected


def test_installed() -> None:
    tested = FakeIrs({})
    original = publication_box_module.urlopen
    with tested.installed() as installed:
        result = (installed is tested, publication_box_module.urlopen is tested)
    expected = (True, True)
    assert result == expected
    exp_restored = original
    assert publication_box_module.urlopen is exp_restored


def test___call__() -> None:
    tested = FakeIrs({URL: b"pdf bytes"})
    request = urllib.request.Request(URL, headers={"User-Agent": "loader"})
    with tested(request, timeout=60.0) as response:
        result = response.read(4)
    expected = b"pdf "
    assert result == expected
    exp_requests = [(URL, "loader", 60.0)]
    assert tested.requests == exp_requests


def test___call____unknown_url_is_404() -> None:
    tested = FakeIrs({})
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        tested("https://www.irs.gov/pub/irs-pdf/nope.pdf")
    result = exc_info.value.code
    expected = 404
    assert result == expected
    # No timeout given: urllib's own default sentinel.
    exp_requests = [("https://www.irs.gov/pub/irs-pdf/nope.pdf", None, socket._GLOBAL_DEFAULT_TIMEOUT)]  # type: ignore[attr-defined]
    assert tested.requests == exp_requests


def test___call____a_failure_is_raised() -> None:
    tested = FakeIrs({URL: TimeoutError("timed out")})
    with pytest.raises(TimeoutError) as exc_info:
        tested(URL, timeout=1.0)
    result = str(exc_info.value)
    expected = "timed out"
    assert result == expected
