from __future__ import annotations

from pydantic import BaseModel

from rag_my_docs.constants.embedding_constants import EmbeddingConstants


class EvaluationRunRequest(BaseModel):
    # The question's place in the test set, from 0.
    index: int
    top_k: int = EmbeddingConstants.default_top_k
