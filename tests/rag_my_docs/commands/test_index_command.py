from __future__ import annotations

import logging
from typing import Any
from unittest.mock import call, patch

import pytest

from rag_my_docs.commands.index_command import IndexCommand
from rag_my_docs.libraries.alerter import Alerter
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.email_box import EmailBox
from rag_my_docs.libraries.embedding_box import EmbeddingBox
from rag_my_docs.libraries.index_store import IndexStore
from rag_my_docs.libraries.job_queue import JobQueue
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.chunk_to_embed import ChunkToEmbed
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.claimed_indexing import ClaimedIndexing
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.settings import Settings
from rag_my_docs.structures.user_account import UserAccount

KEY = "Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI="
MODEL = "BAAI/bge-small-en-v1.5"


def helper_settings() -> Settings:
    return Settings(
        environment="production",
        database_url="postgresql://u:p@h/d",
        app_version="v",
        build_time="b",
        app_commit="c",
        app_base_url="https://docs.example.com",
        data_key=KEY,
        smtp_host="",
        smtp_port=587,
        smtp_user="",
        smtp_password="",
        email_from="noreply@example.com",
        bootstrap_admin_email="",
        chunking=ChunkingSettings(target_tokens=200, max_tokens=350, min_tokens=75, overlap_sentences=1),
        embedding=EmbeddingSettings(model=MODEL, query_prefix="Query: ", batch_size=2),
    )


def helper_instance() -> IndexCommand:
    settings = helper_settings()
    return IndexCommand(
        IndexStore(
            Database(settings.database_url),
            SecretSealer(KEY),
            SectionPaths(Database(settings.database_url), SecretSealer(KEY)),
        ),
        EmbeddingBox(settings.embedding),
        JobQueue("indexing", Alerter(EmailBox(settings), "")),
    )


def helper_claimed() -> ClaimedIndexing:
    return ClaimedIndexing(document_id=5, run=2, model=MODEL)


def helper_chunks(count: int) -> list[ChunkToEmbed]:
    return [ChunkToEmbed(chunk_id=10 + index, text=f"Chunk {index}.") for index in range(count)]


def helper_user() -> UserAccount:
    return UserAccount(
        user_id=7,
        email="ada@example.com",
        display_name="Ada",
        is_disabled=False,
    )


def test___init__() -> None:
    settings = helper_settings()
    store = IndexStore(
        Database(settings.database_url),
        SecretSealer(KEY),
        SectionPaths(Database(settings.database_url), SecretSealer(KEY)),
    )
    box = EmbeddingBox(settings.embedding)
    jobs = JobQueue("indexing", Alerter(EmailBox(settings), ""))
    tested = IndexCommand(store, box, jobs)
    result = (tested._store, tested._box, tested._jobs)
    expected = (store, box, jobs)
    assert result == expected


def test_queued() -> None:
    tested = helper_instance()
    with patch.object(JobQueue, "queued", side_effect=[None]) as mock_queued:
        result = tested.queued(5)
    assert result is None
    exp_calls = [call(tested.indexed, (5,))]
    assert mock_queued.mock_calls == exp_calls


def test_resumed() -> None:
    """The model loads first, on the job thread; then stale runs and documents
    from another model go back to the queue, and everything waiting is queued."""
    tested = helper_instance()
    with (
        patch.object(JobQueue, "queued", side_effect=[None]) as mock_jobs_queued,
        patch.object(IndexStore, "restarted", side_effect=[1]) as mock_restarted,
        patch.object(IndexStore, "outdated", side_effect=[2]) as mock_outdated,
        patch.object(IndexStore, "pending_ids", side_effect=[[3, 4]]) as mock_pending_ids,
        patch.object(IndexCommand, "queued", side_effect=[None, None]) as mock_queued,
    ):
        result = tested.resumed()
    assert result is None
    exp_calls = [call(tested._box.warmed, ())]
    assert mock_jobs_queued.mock_calls == exp_calls
    exp_calls = [call(10)]
    assert mock_restarted.mock_calls == exp_calls
    exp_calls = [call("BAAI/bge-small-en-v1.5+context-v3")]
    assert mock_outdated.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_pending_ids.mock_calls == exp_calls
    exp_calls = [call(3), call(4)]
    assert mock_queued.mock_calls == exp_calls


def test_indexed() -> None:
    tested = helper_instance()
    with (
        patch("rag_my_docs.commands.index_command.time") as mock_time,
        patch.object(IndexStore, "claimed", side_effect=[helper_claimed()]) as mock_claimed,
        patch.object(IndexStore, "missing", side_effect=[helper_chunks(3)]) as mock_missing,
        patch.object(IndexCommand, "embedded", side_effect=[2]) as mock_embedded,
        patch.object(IndexCommand, "closed", side_effect=[None]) as mock_closed,
        patch.object(IndexStore, "failed", side_effect=[]) as mock_failed,
    ):
        mock_time.perf_counter.side_effect = [100.0]
        result = tested.indexed(5)
    assert result is None
    exp_calls = [call.perf_counter()]
    assert mock_time.mock_calls == exp_calls
    exp_calls = [call(5, "BAAI/bge-small-en-v1.5+context-v3")]
    assert mock_claimed.mock_calls == exp_calls
    exp_calls = [call(ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"))]
    assert mock_missing.mock_calls == exp_calls
    exp_calls = [
        call(
            ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"),
            [
                ChunkToEmbed(chunk_id=10, text="Chunk 0."),
                ChunkToEmbed(chunk_id=11, text="Chunk 1."),
                ChunkToEmbed(chunk_id=12, text="Chunk 2."),
            ],
        )
    ]
    assert mock_embedded.mock_calls == exp_calls
    exp_calls = [call(ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"), 3, 2, 100.0)]
    assert mock_closed.mock_calls == exp_calls
    exp_calls = []
    assert mock_failed.mock_calls == exp_calls


def test_indexed__not_waiting() -> None:
    tested = helper_instance()
    with (
        patch.object(IndexStore, "claimed", side_effect=[None]) as mock_claimed,
        patch.object(IndexStore, "missing", side_effect=[]) as mock_missing,
        patch.object(IndexCommand, "closed", side_effect=[]) as mock_closed,
    ):
        result = tested.indexed(5)
    assert result is None
    exp_calls = [call(5, "BAAI/bge-small-en-v1.5+context-v3")]
    assert mock_claimed.mock_calls == exp_calls
    exp_calls = []
    assert mock_missing.mock_calls == exp_calls
    assert mock_closed.mock_calls == exp_calls


def test_indexed__a_known_failure_is_recorded() -> None:
    """The model could not load: the document is failed with the reason, and
    the batches stored before it stay."""
    tested = helper_instance()
    with (
        patch.object(IndexStore, "claimed", side_effect=[helper_claimed()]) as mock_claimed,
        patch.object(IndexStore, "missing", side_effect=[helper_chunks(1)]) as mock_missing,
        patch.object(
            IndexCommand, "embedded", side_effect=[AppException(503, "The embedding model m could not be loaded.")]
        ) as mock_embedded,
        patch.object(IndexStore, "failed", side_effect=[True]) as mock_failed,
        patch.object(IndexCommand, "closed", side_effect=[]) as mock_closed,
    ):
        result = tested.indexed(5)
    assert result is None
    exp_calls = [call(5, "BAAI/bge-small-en-v1.5+context-v3")]
    assert mock_claimed.mock_calls == exp_calls
    exp_calls = [call(ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"))]
    assert mock_missing.mock_calls == exp_calls
    exp_calls = [
        call(
            ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"),
            [ChunkToEmbed(chunk_id=10, text="Chunk 0.")],
        )
    ]
    assert mock_embedded.mock_calls == exp_calls
    exp_calls = [
        call(
            ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"),
            "The embedding model m could not be loaded.",
            {},
        )
    ]
    assert mock_failed.mock_calls == exp_calls
    exp_calls = []
    assert mock_closed.mock_calls == exp_calls


def test_indexed__unexpected_failure(caplog: pytest.LogCaptureFixture) -> None:
    tested = helper_instance()
    with (
        patch.object(IndexStore, "claimed", side_effect=[helper_claimed()]) as mock_claimed,
        patch.object(IndexStore, "missing", side_effect=[ValueError("boom")]) as mock_missing,
        patch.object(IndexStore, "failed", side_effect=[True]) as mock_failed,
        pytest.raises(ValueError) as exc_info,
    ):
        tested.indexed(5)
    assert str(exc_info.value) == "boom"
    exp_calls = [call(5, "BAAI/bge-small-en-v1.5+context-v3")]
    assert mock_claimed.mock_calls == exp_calls
    exp_calls = [call(ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"))]
    assert mock_missing.mock_calls == exp_calls
    exp_calls = [
        call(
            ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"),
            "Indexing failed unexpectedly. Try again, or report it.",
            {},
        )
    ]
    assert mock_failed.mock_calls == exp_calls
    exp_messages = ["[INDEXING] document 5 failed"]
    assert [record.getMessage() for record in caplog.records] == exp_messages


def test_embedded() -> None:
    """Three chunks with a batch size of two: two batches, each stored as it comes."""
    tested = helper_instance()
    with (
        patch.object(EmbeddingBox, "passages", side_effect=[[[0.1], [0.2]], [[0.3]]]) as mock_passages,
        patch.object(IndexStore, "embedded", side_effect=[None, None]) as mock_embedded,
    ):
        result = tested.embedded(helper_claimed(), helper_chunks(3))
    expected = 2
    assert result == expected
    exp_calls = [call(["Chunk 0.", "Chunk 1."]), call(["Chunk 2."])]
    assert mock_passages.mock_calls == exp_calls
    exp_calls = [
        call(
            ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"),
            [ChunkToEmbed(chunk_id=10, text="Chunk 0."), ChunkToEmbed(chunk_id=11, text="Chunk 1.")],
            [[0.1], [0.2]],
        ),
        call(
            ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"),
            [ChunkToEmbed(chunk_id=12, text="Chunk 2.")],
            [[0.3]],
        ),
    ]
    assert mock_embedded.mock_calls == exp_calls


@pytest.mark.parametrize(
    "counts, exp_failed, exp_finished, exp_messages",
    [
        pytest.param(
            (3, 3),
            [],
            [
                call(
                    ClaimedIndexing(5, 2, MODEL),
                    {"chunks": 3, "embedded": 2, "reused": 1, "batches": 1, "duration_ms": 250.0},
                )
            ],
            ["[INDEXING] document 5: 3 chunks, 2 embedded in 250.0 ms"],
            id="every-chunk-embedded",
        ),
        pytest.param(
            (0, 0),
            [
                call(
                    ClaimedIndexing(5, 2, MODEL),
                    "This document has no chunks to index: process it first.",
                    {"chunks": 0, "embedded": 2, "reused": -2, "batches": 1, "duration_ms": 250.0},
                )
            ],
            [],
            [],
            id="no-chunks-left",
        ),
        pytest.param(
            (3, 2),
            [
                call(
                    ClaimedIndexing(5, 2, MODEL),
                    "Some chunks have no embedding after indexing.",
                    {"chunks": 3, "embedded": 2, "reused": 1, "batches": 1, "duration_ms": 250.0},
                )
            ],
            [],
            [],
            id="incomplete",
        ),
    ],
)
def test_closed(
    caplog: pytest.LogCaptureFixture,
    counts: tuple[int, int],
    exp_failed: list[Any],
    exp_finished: list[Any],
    exp_messages: list[str],
) -> None:
    caplog.set_level(logging.INFO, logger="rag_my_docs")
    tested = helper_instance()
    with (
        patch("rag_my_docs.commands.index_command.time") as mock_time,
        patch.object(IndexStore, "counts", side_effect=[counts]) as mock_counts,
        patch.object(IndexStore, "failed", side_effect=[True] * len(exp_failed)) as mock_failed,
        patch.object(IndexStore, "finished", side_effect=[True] * len(exp_finished)) as mock_finished,
    ):
        mock_time.perf_counter.side_effect = [100.25]
        result = tested.closed(helper_claimed(), 2, 1, 100.0)
    assert result is None
    exp_calls = [call.perf_counter()]
    assert mock_time.mock_calls == exp_calls
    exp_calls = [call(ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"))]
    assert mock_counts.mock_calls == exp_calls
    assert mock_failed.mock_calls == exp_failed
    assert mock_finished.mock_calls == exp_finished
    assert [record.getMessage() for record in caplog.records] == exp_messages


def test_closed__a_run_that_lost_the_document_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="rag_my_docs")
    tested = helper_instance()
    with (
        patch("rag_my_docs.commands.index_command.time") as mock_time,
        patch.object(IndexStore, "counts", side_effect=[(1, 1)]) as mock_counts,
        patch.object(IndexStore, "finished", side_effect=[False]) as mock_finished,
    ):
        mock_time.perf_counter.side_effect = [101.0]
        result = tested.closed(helper_claimed(), 1, 1, 100.0)
    assert result is None
    exp_calls = [call.perf_counter()]
    assert mock_time.mock_calls == exp_calls
    exp_calls = [call(ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"))]
    assert mock_counts.mock_calls == exp_calls
    exp_calls = [
        call(
            ClaimedIndexing(document_id=5, run=2, model="BAAI/bge-small-en-v1.5"),
            {"chunks": 1, "embedded": 1, "reused": 0, "batches": 1, "duration_ms": 1000.0},
        )
    ]
    assert mock_finished.mock_calls == exp_calls
    exp_messages: list[str] = []
    assert [record.getMessage() for record in caplog.records] == exp_messages


@pytest.mark.parametrize(
    "count, size, expected",
    [
        pytest.param(5, 2, [[10, 11], [12, 13], [14]], id="last-batch-short"),
        pytest.param(4, 2, [[10, 11], [12, 13]], id="exact"),
        pytest.param(1, 32, [[10]], id="one"),
        pytest.param(0, 32, [], id="none"),
    ],
)
def test_batches(count: int, size: int, expected: list[list[int]]) -> None:
    tested = IndexCommand
    result = tested.batches(helper_chunks(count), size)
    assert [[chunk.chunk_id for chunk in batch] for batch in result] == expected


def test_reindexed() -> None:
    tested = helper_instance()
    with (
        patch.object(IndexStore, "requested", side_effect=[1]) as mock_requested,
        patch.object(IndexStore, "exists", side_effect=[]) as mock_exists,
        patch.object(IndexCommand, "queued", side_effect=[None]) as mock_queued,
    ):
        result = tested.reindexed(helper_user(), 5, True)
    assert result is None
    exp_calls = [call(7, 5, True)]
    assert mock_requested.mock_calls == exp_calls
    exp_calls = []
    assert mock_exists.mock_calls == exp_calls
    exp_calls = [call(5)]
    assert mock_queued.mock_calls == exp_calls


@pytest.mark.parametrize(
    "exists, expected",
    [
        pytest.param(False, (404, "This document does not exist."), id="not-the-owners"),
        pytest.param(True, (400, "This document has no chunks to index: process it first."), id="not-processed"),
    ],
)
def test_reindexed__refused(exists: bool, expected: tuple[int, str]) -> None:
    tested = helper_instance()
    with (
        patch.object(IndexStore, "requested", side_effect=[0]) as mock_requested,
        patch.object(IndexStore, "exists", side_effect=[exists]) as mock_exists,
        patch.object(IndexCommand, "queued", side_effect=[]) as mock_queued,
        pytest.raises(AppException) as exc_info,
    ):
        tested.reindexed(helper_user(), 5, False)
    assert (exc_info.value.status_code, exc_info.value.message) == expected
    exp_calls = [call(7, 5, False)]
    assert mock_requested.mock_calls == exp_calls
    exp_calls = [call(7, 5)]
    assert mock_exists.mock_calls == exp_calls
    exp_calls = []
    assert mock_queued.mock_calls == exp_calls


def test_indexed_all() -> None:
    tested = helper_instance()
    with (
        patch.object(IndexStore, "requested_all", side_effect=[[3, 4]]) as mock_requested_all,
        patch.object(IndexCommand, "queued", side_effect=[None, None]) as mock_queued,
    ):
        result = tested.indexed_all(helper_user())
    expected = 2
    assert result == expected
    exp_calls = [call(7, "BAAI/bge-small-en-v1.5+context-v3")]
    assert mock_requested_all.mock_calls == exp_calls
    exp_calls = [call(3), call(4)]
    assert mock_queued.mock_calls == exp_calls
