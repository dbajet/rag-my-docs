from __future__ import annotations

import urllib.error

import pytest

from rag_my_docs.libraries.publication_box import PublicationBox
from rag_my_docs.structures.app_exception import AppException
from tests.support.fake_irs import FakeIrs

URL = "https://www.irs.gov/pub/irs-pdf/p17.pdf"


def test_downloaded() -> None:
    tested = PublicationBox()
    fake = FakeIrs({URL: b"%PDF-1.7 publication"})
    with fake.installed():
        result = tested.downloaded(URL)
    expected = b"%PDF-1.7 publication"
    assert result == expected
    exp_requests = [(URL, "rag_my_docs corpus loader", 60.0)]
    assert fake.requests == exp_requests


@pytest.mark.parametrize(
    "files",
    [
        pytest.param({}, id="not-found"),
        pytest.param({URL: urllib.error.URLError("no route")}, id="unreachable"),
        pytest.param({URL: TimeoutError("timed out")}, id="timeout"),
    ],
)
def test_downloaded__failed(files: dict[str, bytes | BaseException]) -> None:
    tested = PublicationBox()
    fake = FakeIrs(files)
    with fake.installed(), pytest.raises(AppException) as exc_info:
        tested.downloaded(URL)
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (502, "The publication could not be downloaded.")
    assert result == expected
    exp_requests = [(URL, "rag_my_docs corpus loader", 60.0)]
    assert fake.requests == exp_requests


def test_downloaded__too_large() -> None:
    tested = PublicationBox()
    fake = FakeIrs({URL: b"x" * (30 * 1024 * 1024 + 1)})
    with fake.installed(), pytest.raises(AppException) as exc_info:
        tested.downloaded(URL)
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (502, "The publication is larger than any publication should be.")
    assert result == expected
