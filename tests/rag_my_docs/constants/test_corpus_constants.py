from __future__ import annotations

import dataclasses

from rag_my_docs.constants.corpus_constants import CorpusConstants
from tests.conftest import is_dataclass


def test_class() -> None:
    tested = CorpusConstants
    fields = {
        "publications": "tuple[tuple[str, str, str], ...]",
        "download_timeout_seconds": "float",
        "max_download_bytes": "int",
        "user_agent": "str",
        "download_failed": "str",
        "download_too_large": "str",
    }
    assert is_dataclass(tested, fields)


def test_values() -> None:
    tested = CorpusConstants
    result = dataclasses.asdict(tested)
    expected = {
        "publications": (
            (
                "Publication 17 — Your Federal Income Tax",
                "https://www.irs.gov/pub/irs-pdf/p17.pdf",
                "2d2381d62c7c77ed4b64b83ac23c28ed85b7c8304eb580c09bd9fd660be0840b",
            ),
            (
                "Publication 334 — Tax Guide for Small Business",
                "https://www.irs.gov/pub/irs-pdf/p334.pdf",
                "722639ba7bfd6f5348234b58d21e1afd8cb0c04895138bd5c4539142dde6c60a",
            ),
            (
                "Publication 463 — Travel, Gift, and Car Expenses",
                "https://www.irs.gov/pub/irs-pdf/p463.pdf",
                "ddc3bb15e437bd65c37da2036debdae78c1a0bf5a080b19d5301c81f5ec5ae5d",
            ),
            (
                "Publication 501 — Dependents, Standard Deduction, and Filing Information",
                "https://www.irs.gov/pub/irs-pdf/p501.pdf",
                "288b62a4917f3bc3df0bade7a22c23a8ae648ba9eb5086df5b8b50e881148ff6",
            ),
            (
                "Publication 502 — Medical and Dental Expenses",
                "https://www.irs.gov/pub/irs-pdf/p502.pdf",
                "11801353ab959ee04dcadc1f799b6c0b87e1879680fe6f8511af3fa113690d25",
            ),
            (
                "Publication 503 — Child and Dependent Care Expenses",
                "https://www.irs.gov/pub/irs-pdf/p503.pdf",
                "3a00d0c1a1d6d3a88afdf418b79873f95f6d2de584c330ec67647139c09c5532",
            ),
            (
                "Publication 526 — Charitable Contributions",
                "https://www.irs.gov/pub/irs-pdf/p526.pdf",
                "607011d9065a39b2fa3e08eda90c6313c581c2d4515f00fc23b0ae0a145435ae",
            ),
            (
                "Publication 527 — Residential Rental Property",
                "https://www.irs.gov/pub/irs-pdf/p527.pdf",
                "20c826e98dce23d20f80ab89da7e668a8ee7162a2194f480ce417cff651201d6",
            ),
            (
                "Publication 587 — Business Use of Your Home",
                "https://www.irs.gov/pub/irs-pdf/p587.pdf",
                "7f68de213b25a0f826b6112191fcf89e29c121865ef59c4945f8699de0a6d2af",
            ),
            (
                "Publication 590-A — Contributions to Individual Retirement Arrangements (IRAs)",
                "https://www.irs.gov/pub/irs-pdf/p590a.pdf",
                "ca0e42873cfd66d2d9bfd082125d381f8a0c2f3b0a441d7d65ab5ccc06a0e1c7",
            ),
            (
                "Publication 970 — Tax Benefits for Education",
                "https://www.irs.gov/pub/irs-pdf/p970.pdf",
                "167f3d6df0d9370a3980b25b7a2e7b479ab7f336cce763d1f0c1c4962b18285e",
            ),
            (
                "Publication 1040 — Tax and Earned Income Credit Tables",
                "https://www.irs.gov/pub/irs-pdf/i1040tt.pdf",
                "8ae019bd3b28b07b37e46e18597d8e899222a2224ce7482c015b130033403532",
            ),
        ),
        "download_timeout_seconds": 60.0,
        "max_download_bytes": 31457280,
        "user_agent": "rag_my_docs corpus loader",
        "download_failed": "The publication could not be downloaded.",
        "download_too_large": "The publication is larger than any publication should be.",
    }
    assert result == expected
