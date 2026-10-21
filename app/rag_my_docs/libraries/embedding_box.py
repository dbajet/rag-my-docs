from __future__ import annotations

import threading
from typing import Any

from sentence_transformers import SentenceTransformer

from rag_my_docs.constants.embedding_constants import EmbeddingConstants
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.embedding_settings import EmbeddingSettings


class EmbeddingBox:
    """The local embedding model (Sentence Transformers), loaded once per
    process and shared: by the indexing thread, which embeds passages in
    batches, and by searches, which embed one question each.

    Everything runs in this process: in production the weights are baked into
    the image and the Hugging Face libraries are told to stay offline
    (Dockerfile), so no text ever leaves the server. Vectors are normalised to
    length 1, so cosine similarity is a plain dot product, for passages and
    questions alike. Following the model's documented convention, a question
    carries the retrieval instruction and a passage does not.

    One lock covers loading and inference: two threads never load the model
    twice, nor run it at the same time."""

    # Sentence Transformers' constructor is reached through this one reference,
    # typed as Any; everything it returns is turned into plain floats here.
    _LIBRARY: Any = SentenceTransformer

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._model: Any = None
        self._lock = threading.Lock()

    def space(self) -> str:
        """What a stored vector is comparable with: vectors from the same model,
        indexed the same way. Stored beside every vector."""
        return f"{self._settings.model}+{EmbeddingConstants.index_format}"

    def batch_size(self) -> int:
        return self._settings.batch_size

    def warmed(self) -> None:
        """Loads the model ahead of the first request: run on a job thread at startup."""
        with self._lock:
            self._loaded()

    def passages(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return self._encoded(self._loaded(), texts)

    def query(self, text: str) -> list[float]:
        with self._lock:
            return self._encoded(self._loaded(), [self._settings.query_prefix + text])[0]

    def _loaded(self) -> Any:
        """The model, loaded on first use; the lock is held by the caller."""
        if self._model is not None:
            return self._model
        try:
            model = self._LIBRARY(self._settings.model, device=EmbeddingConstants.device)
        except (OSError, ValueError) as error:
            raise AppException(503, f"The embedding model {self._settings.model} could not be loaded.") from error
        dimension = int(model.get_embedding_dimension())
        if dimension != EmbeddingConstants.dimension:
            raise AppException(
                500,
                f"The embedding model {self._settings.model} makes vectors of {dimension} numbers; "
                f"the database stores {EmbeddingConstants.dimension}.",
            )
        self._model = model
        return model

    def _encoded(self, model: Any, texts: list[str]) -> list[list[float]]:
        try:
            vectors = model.encode(
                texts,
                batch_size=self._settings.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except (RuntimeError, ValueError) as error:
            raise AppException(500, "The embedding model could not encode the text.") from error
        return [[float(value) for value in vector] for vector in vectors]
