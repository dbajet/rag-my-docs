from __future__ import annotations

from pydantic import BaseModel

from rag_my_docs.constants.embedding_constants import EmbeddingConstants


class AnswerRequest(BaseModel):
    question: str
    # How many passages the model is given.
    top_k: int = EmbeddingConstants.default_top_k
