from __future__ import annotations

import hashlib
import hmac

from rag_my_docs.constants.constants import Constants
from rag_my_docs.constants.embedding_constants import EmbeddingConstants


class TextDigest:
    """A keyed fingerprint of a chunk's passage (`PassageText`: its text and
    the section path above it), stored beside it: when a document is processed
    again, a chunk whose passage did not change keeps its embedding instead of
    being embedded again.

    Keyed, like the email blind index: a plain hash of a sealed text would let
    anyone holding a dump test guesses against it. The key is derived from the
    data key under a label of its own, so it is never the sealing key itself."""

    def __init__(self, data_key: str) -> None:
        self._key = hmac.new(
            data_key.encode(Constants.text_encoding),
            EmbeddingConstants.digest_label.encode(Constants.text_encoding),
            hashlib.sha256,
        ).digest()

    def digest(self, text: str) -> str:
        return hmac.new(self._key, text.encode(Constants.text_encoding), hashlib.sha256).hexdigest()
