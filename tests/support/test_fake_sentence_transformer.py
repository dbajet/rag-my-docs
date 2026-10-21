from __future__ import annotations

import pytest

from rag_my_docs.libraries.embedding_box import EmbeddingBox
from tests.support.fake_sentence_transformer import FakeSentenceTransformer


def test___init__() -> None:
    tested = FakeSentenceTransformer(dimension=4, failing_encode=True, unknown=True)
    result = (tested.dimension, tested.failing_encode, tested.unknown, tested.calls)
    expected = (4, True, True, [])
    assert result == expected


def test_installed() -> None:
    tested = FakeSentenceTransformer()
    original = EmbeddingBox._LIBRARY
    with tested.installed() as installed:
        result = (installed is tested, EmbeddingBox._LIBRARY is tested)
    expected = (True, True)
    assert result == expected
    exp_restored = original
    assert EmbeddingBox._LIBRARY is exp_restored


def test___call__() -> None:
    tested = FakeSentenceTransformer()
    result = tested("BAAI/bge-small-en-v1.5", device="cpu")
    assert result is tested
    expected = [("load", "BAAI/bge-small-en-v1.5", "cpu")]
    assert tested.calls == expected


def test___call____an_unknown_model() -> None:
    tested = FakeSentenceTransformer(unknown=True)
    with pytest.raises(OSError) as exc_info:
        tested("no/such-model", device="cpu")
    exp_message = "no/such-model is not a local folder and is not a valid model identifier"
    assert str(exc_info.value) == exp_message
    expected = [("load", "no/such-model", "cpu")]
    assert tested.calls == expected


def test_get_embedding_dimension() -> None:
    tested = FakeSentenceTransformer(dimension=4)
    result = tested.get_embedding_dimension()
    expected = 4
    assert result == expected
    exp_calls = [("get_embedding_dimension",)]
    assert tested.calls == exp_calls


@pytest.mark.parametrize(
    "sentences, normalize, expected, exp_calls",
    [
        # With 4 dimensions, md5 puts "sales" in 0 and "grew" in 3 (computed with md5sum).
        pytest.param(
            ["Sales grew grew", ""],
            False,
            [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
            [("encode", ("Sales grew grew", ""), 8, False)],
            id="counts",
        ),
        pytest.param(
            ["Sales grew grew", ""],
            True,
            [[0.447214, 0.0, 0.0, 0.894427], [0.0, 0.0, 0.0, 0.0]],
            [("encode", ("Sales grew grew", ""), 8, True)],
            id="normalised",
        ),
        pytest.param(
            "GREW", True, [[0.0, 0.0, 0.0, 1.0]], [("encode", ("GREW",), 8, True)], id="one-string-lowercased"
        ),
    ],
)
def test_encode(
    sentences: str | list[str],
    normalize: bool,
    expected: list[list[float]],
    exp_calls: list[tuple[object, ...]],
) -> None:
    tested = FakeSentenceTransformer(dimension=4)
    result = tested.encode(sentences, batch_size=8, normalize_embeddings=normalize, convert_to_numpy=True)
    assert [[round(float(value), 6) for value in row] for row in result] == expected
    assert tested.calls == exp_calls


def test_encode__failing() -> None:
    tested = FakeSentenceTransformer(failing_encode=True)
    with pytest.raises(RuntimeError) as exc_info:
        tested.encode(["a"], batch_size=8, normalize_embeddings=True)
    assert str(exc_info.value) == "Inference failed"
    expected = [("encode", ("a",), 8, True)]
    assert tested.calls == expected


@pytest.mark.parametrize(
    "word, expected",
    [
        pytest.param("grew", 183, id="grew"),
        pytest.param("sales", 152, id="sales"),
    ],
)
def test_bucket(word: str, expected: int) -> None:
    tested = FakeSentenceTransformer()
    result = tested.bucket(word)
    assert result == expected
