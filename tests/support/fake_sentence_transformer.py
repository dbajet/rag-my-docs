from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np

from rag_my_docs.libraries.embedding_box import EmbeddingBox


class FakeSentenceTransformer:
    """A hand-written stand-in for `sentence_transformers.SentenceTransformer`:
    no weights, no PyTorch run, the same answer every time.

    It reproduces the shape `EmbeddingBox` meets: the class is called with
    `(model_name_or_path, device=...)` to load a model, `get_embedding_dimension()`
    gives the vector size, and `encode(sentences, batch_size=..., normalize_embeddings=...,
    convert_to_numpy=True, show_progress_bar=...)` returns a numpy array with one
    row per sentence, of length 1 when normalised.

    Its vectors are a bag of words: each lowercased word adds 1 to one of the
    dimensions, chosen by a hash of the word. Texts that share words point in
    similar directions, so ranking tests have meaning without a real model.

    It fails the way the library does, on demand: loading an unknown model
    raises `OSError`; `failing_encode` makes `encode` raise `RuntimeError`.
    Every call is recorded in order in `calls`."""

    _TARGET = "_LIBRARY"
    _WORD = re.compile(r"\w+")

    def __init__(self, dimension: int = 384, failing_encode: bool = False, unknown: bool = False) -> None:
        self.dimension = dimension
        self.failing_encode = failing_encode
        self.unknown = unknown
        self.calls: list[tuple[Any, ...]] = []

    @contextmanager
    def installed(self) -> Iterator[FakeSentenceTransformer]:
        """In place of the class `EmbeddingBox` constructs, for the block."""
        with patch.object(EmbeddingBox, self._TARGET, new=self):
            yield self

    def __call__(
        self, model_name_or_path: str | None = None, device: str | None = None, **kwargs: Any
    ) -> FakeSentenceTransformer:
        self.calls.append(("load", model_name_or_path, device))
        if self.unknown:
            raise OSError(f"{model_name_or_path} is not a local folder and is not a valid model identifier")
        return self

    def get_embedding_dimension(self) -> int:
        self.calls.append(("get_embedding_dimension",))
        return self.dimension

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        **kwargs: Any,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        texts = [sentences] if isinstance(sentences, str) else list(sentences)
        self.calls.append(("encode", tuple(texts), batch_size, normalize_embeddings))
        if self.failing_encode:
            raise RuntimeError("Inference failed")
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in self._WORD.findall(text.lower()):
                result[row, self.bucket(word)] += 1.0
            norm = float(np.linalg.norm(result[row]))
            if normalize_embeddings and norm > 0:
                result[row] /= norm
        return result

    def bucket(self, word: str) -> int:
        return int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dimension
