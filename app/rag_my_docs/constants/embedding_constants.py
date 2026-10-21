from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _EmbeddingConstants:
    # --- the model (each overridable from the environment) ---------------------
    env_model: str = "EMBEDDING_MODEL"
    env_query_prefix: str = "EMBEDDING_QUERY_PREFIX"
    env_batch_size: str = "EMBEDDING_BATCH_SIZE"
    default_model: str = "BAAI/bge-small-en-v1.5"
    # BGE's documented instruction for a short question searched against passages;
    # passages are encoded as they are, without any prefix.
    default_query_prefix: str = "Represent this sentence for searching relevant passages: "
    default_batch_size: int = 32
    # The size of `document_chunks.embedding`: a model of any other size is refused.
    dimension: int = 384
    # Inference runs on the CPU: the server has no GPU.
    device: str = "cpu"
    # How a chunk is indexed: its passage embedded (`PassageText`). Stored
    # with every vector beside the model's name, so changing either one makes
    # the index stale, and every chunk is indexed again at the next start.
    # v3: words are no longer hashed beside the vector.
    index_format: str = "context-v3"

    # --- search -----------------------------------------------------------------
    default_top_k: int = 5
    max_top_k: int = 50
    max_query_chars: int = 1000
    # Search results name a section by its path: "2. Results › 2.1 Details".
    path_separator: str = " › "

    # --- indexing ---------------------------------------------------------------
    # An indexing run started this long ago on a color that is gone is started again.
    stale_run_minutes: int = 10
    # The HMAC label deriving the chunk-digest key from the data key.
    digest_label: str = "rag_my_docs chunk text digest v1"

    # --- messages ---------------------------------------------------------------
    nothing_indexed: str = "No publication is indexed yet: they are downloaded and indexed after the app starts."
    not_indexable: str = "This document has no chunks to index: process it first."
    unexpected: str = "Indexing failed unexpectedly. Try again, or report it."
    incomplete: str = "Some chunks have no embedding after indexing."


EmbeddingConstants = _EmbeddingConstants()
