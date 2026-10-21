from __future__ import annotations

import urllib.error
from urllib.request import Request, urlopen

from rag_my_docs.constants.corpus_constants import CorpusConstants
from rag_my_docs.structures.app_exception import AppException


class PublicationBox:
    """The IRS file server: a publication's PDF, by its URL. `urlopen` is
    imported by name so a test replaces this download alone, not the
    language models' HTTP calls beside it."""

    def downloaded(self, url: str) -> bytes:
        request = Request(url, headers={"User-Agent": CorpusConstants.user_agent})
        try:
            with urlopen(request, timeout=CorpusConstants.download_timeout_seconds) as response:
                result: bytes = response.read(CorpusConstants.max_download_bytes + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise AppException(502, CorpusConstants.download_failed) from error
        if len(result) > CorpusConstants.max_download_bytes:
            raise AppException(502, CorpusConstants.download_too_large)
        return result
