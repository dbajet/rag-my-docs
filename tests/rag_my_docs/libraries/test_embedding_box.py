from __future__ import annotations

import pytest

from rag_my_docs.libraries.embedding_box import EmbeddingBox
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from tests.support.fake_sentence_transformer import FakeSentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"


def helper_settings() -> EmbeddingSettings:
    # The fake makes a bag of words: this prefix doubles the question's own word.
    return EmbeddingSettings(model=MODEL, query_prefix="grew ", batch_size=8)


def helper_axes(values: dict[int, float]) -> list[float]:
    """A 384-number vector, zero but at the given positions. In the fake, md5
    puts "sales" in 152 and "grew" in 183 (computed with md5sum)."""
    return [values.get(index, 0.0) for index in range(384)]


def helper_rounded(vectors: list[list[float]]) -> list[list[float]]:
    return [[round(value, 6) for value in vector] for vector in vectors]


def test___init__() -> None:
    settings = helper_settings()
    tested = EmbeddingBox(settings)
    result = (tested._settings, tested._model, tested._lock.locked())
    expected = (settings, None, False)
    assert result == expected


def test_space() -> None:
    tested = EmbeddingBox(helper_settings())
    result = tested.space()
    expected = "BAAI/bge-small-en-v1.5+context-v3"
    assert result == expected


def test_batch_size() -> None:
    tested = EmbeddingBox(helper_settings())
    result = tested.batch_size()
    expected = 8
    assert result == expected


def test_warmed() -> None:
    tested = EmbeddingBox(helper_settings())
    fake = FakeSentenceTransformer()
    with fake.installed():
        result = tested.warmed()
    assert result is None
    assert tested._model is fake
    expected = [("load", MODEL, "cpu"), ("get_embedding_dimension",)]
    assert fake.calls == expected


def test_passages() -> None:
    """Passages are encoded as they are, normalised, in the configured batch size."""
    tested = EmbeddingBox(helper_settings())
    fake = FakeSentenceTransformer()
    with fake.installed():
        result = tested.passages(["Sales grew", "grew"])
    expected = [helper_axes({152: 0.707107, 183: 0.707107}), helper_axes({183: 1.0})]
    assert helper_rounded(result) == expected
    exp_calls = [
        ("load", MODEL, "cpu"),
        ("get_embedding_dimension",),
        ("encode", ("Sales grew", "grew"), 8, True),
    ]
    assert fake.calls == exp_calls


def test_query() -> None:
    """A question carries the query prefix; the model is loaded once and kept."""
    tested = EmbeddingBox(helper_settings())
    fake = FakeSentenceTransformer()
    with fake.installed():
        result = [tested.query("grew"), tested.query("sales")]
    expected = [helper_axes({183: 1.0}), helper_axes({152: 0.707107, 183: 0.707107})]
    assert helper_rounded(result) == expected
    exp_calls = [
        ("load", MODEL, "cpu"),
        ("get_embedding_dimension",),
        ("encode", ("grew grew",), 8, True),
        ("encode", ("grew sales",), 8, True),
    ]
    assert fake.calls == exp_calls


def test__loaded() -> None:
    tested = EmbeddingBox(helper_settings())
    fake = FakeSentenceTransformer()
    with fake.installed():
        result = (tested._loaded(), tested._loaded())
    expected = (fake, fake)
    assert result == expected
    exp_calls = [("load", MODEL, "cpu"), ("get_embedding_dimension",)]
    assert fake.calls == exp_calls


def test__loaded__an_unknown_model_is_unavailable() -> None:
    tested = EmbeddingBox(helper_settings())
    fake = FakeSentenceTransformer(unknown=True)
    with fake.installed(), pytest.raises(AppException) as exc_info:
        tested._loaded()
    result = (exc_info.value.status_code, exc_info.value.message, tested._model)
    expected = (503, "The embedding model BAAI/bge-small-en-v1.5 could not be loaded.", None)
    assert result == expected
    exp_calls = [("load", MODEL, "cpu")]
    assert fake.calls == exp_calls


def test__loaded__another_dimension_is_refused() -> None:
    tested = EmbeddingBox(helper_settings())
    fake = FakeSentenceTransformer(dimension=768)
    with fake.installed(), pytest.raises(AppException) as exc_info:
        tested._loaded()
    result = (exc_info.value.status_code, exc_info.value.message, tested._model)
    expected = (
        500,
        "The embedding model BAAI/bge-small-en-v1.5 makes vectors of 768 numbers; the database stores 384.",
        None,
    )
    assert result == expected
    exp_calls = [("load", MODEL, "cpu"), ("get_embedding_dimension",)]
    assert fake.calls == exp_calls


def test__encoded() -> None:
    tested = EmbeddingBox(helper_settings())
    fake = FakeSentenceTransformer()
    result = tested._encoded(fake, ["grew"])
    expected = [helper_axes({183: 1.0})]
    assert helper_rounded(result) == expected
    assert all(type(value) is float for value in result[0])
    exp_calls = [("encode", ("grew",), 8, True)]
    assert fake.calls == exp_calls


def test__encoded__a_failure_is_an_error() -> None:
    tested = EmbeddingBox(helper_settings())
    fake = FakeSentenceTransformer(failing_encode=True)
    with pytest.raises(AppException) as exc_info:
        tested._encoded(fake, ["grew"])
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (500, "The embedding model could not encode the text.")
    assert result == expected
    exp_calls = [("encode", ("grew",), 8, True)]
    assert fake.calls == exp_calls
