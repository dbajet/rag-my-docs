from __future__ import annotations

import dataclasses

from rag_my_docs.constants.embedding_constants import EmbeddingConstants
from tests.conftest import is_dataclass


def test_class() -> None:
    tested = EmbeddingConstants
    fields = {
        "env_model": "str",
        "env_query_prefix": "str",
        "env_batch_size": "str",
        "default_model": "str",
        "default_query_prefix": "str",
        "default_batch_size": "int",
        "dimension": "int",
        "device": "str",
        "index_format": "str",
        "default_top_k": "int",
        "max_top_k": "int",
        "max_query_chars": "int",
        "path_separator": "str",
        "stale_run_minutes": "int",
        "digest_label": "str",
        "nothing_indexed": "str",
        "not_indexable": "str",
        "unexpected": "str",
        "incomplete": "str",
    }
    assert is_dataclass(tested, fields)


def test_values() -> None:
    tested = EmbeddingConstants
    result = dataclasses.asdict(tested)
    expected = {
        "env_model": "EMBEDDING_MODEL",
        "env_query_prefix": "EMBEDDING_QUERY_PREFIX",
        "env_batch_size": "EMBEDDING_BATCH_SIZE",
        "default_model": "BAAI/bge-small-en-v1.5",
        "default_query_prefix": "Represent this sentence for searching relevant passages: ",
        "default_batch_size": 32,
        "dimension": 384,
        "device": "cpu",
        "index_format": "context-v3",
        "default_top_k": 5,
        "max_top_k": 50,
        "max_query_chars": 1000,
        "path_separator": " › ",
        "stale_run_minutes": 10,
        "digest_label": "rag_my_docs chunk text digest v1",
        "nothing_indexed": "No publication is indexed yet: they are downloaded and indexed after the app starts.",
        "not_indexable": "This document has no chunks to index: process it first.",
        "unexpected": "Indexing failed unexpectedly. Try again, or report it.",
        "incomplete": "Some chunks have no embedding after indexing.",
    }
    assert result == expected
