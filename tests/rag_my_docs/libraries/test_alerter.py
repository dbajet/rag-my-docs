from __future__ import annotations

import logging
import threading
from unittest.mock import call, patch

import pytest

from rag_my_docs.libraries import alerter as alerter_module
from rag_my_docs.libraries.alerter import Alerter
from rag_my_docs.libraries.email_box import EmailBox
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.settings import Settings


def helper_email_box() -> EmailBox:
    return EmailBox(
        Settings(
            environment="production",
            database_url="postgresql://u:p@h/d",
            app_version="2026-10-11.1",
            build_time="20261011-120000Z",
            app_commit="abc123",
            app_base_url="https://docs.example.com",
            data_key="the-data-key",
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_user="",
            smtp_password="",
            email_from="noreply@example.com",
            bootstrap_admin_email="",
            chunking=ChunkingSettings(target_tokens=200, max_tokens=350, min_tokens=75, overlap_sentences=1),
            embedding=EmbeddingSettings(
                model="BAAI/bge-small-en-v1.5",
                query_prefix="Represent this sentence for searching relevant passages: ",
                batch_size=32,
            ),
        )
    )


def helper_instance(recipient: str = "ops@example.com") -> Alerter:
    return Alerter(helper_email_box(), recipient)


def test___init__() -> None:
    email_box = helper_email_box()
    tested = Alerter(email_box, "ops@example.com")
    result = (tested._email_box is email_box, tested._recipient, tested._last_sent, type(tested._lock))
    expected = (True, "ops@example.com", {}, type(threading.Lock()))
    assert result == expected


def test_alerted() -> None:
    tested = helper_instance()
    with (
        patch.object(Alerter, "_is_due", side_effect=[True]) as mock_is_due,
        patch.object(EmailBox, "send", side_effect=[None]) as mock_send,
    ):
        result = tested.alerted("emails", "SMTPDataError")
    assert result is None
    exp_calls = [call("emails")]
    assert mock_is_due.mock_calls == exp_calls
    exp_calls = [
        call(
            "ops@example.com",
            "[RAG My Docs] a background job failed",
            [
                "The background job 'emails' failed: SMTPDataError.",
                "The details are in the logs: docker logs rag_my_docs-app-<color>.",
            ],
        )
    ]
    assert mock_send.mock_calls == exp_calls


def test_alerted__no_recipient() -> None:
    tested = helper_instance(recipient="")
    with (
        patch.object(Alerter, "_is_due", side_effect=[True]) as mock_is_due,
        patch.object(EmailBox, "send", side_effect=[None]) as mock_send,
    ):
        tested.alerted("emails", "SMTPDataError")
    exp_calls: list[object] = []
    assert mock_is_due.mock_calls == exp_calls
    assert mock_send.mock_calls == exp_calls


def test_alerted__not_due() -> None:
    tested = helper_instance()
    with (
        patch.object(Alerter, "_is_due", side_effect=[False]) as mock_is_due,
        patch.object(EmailBox, "send", side_effect=[None]) as mock_send,
    ):
        tested.alerted("emails", "SMTPDataError")
    exp_calls = [call("emails")]
    assert mock_is_due.mock_calls == exp_calls
    exp_calls = []
    assert mock_send.mock_calls == exp_calls


def test_alerted__second_within_the_interval_is_skipped() -> None:
    tested = helper_instance()
    with (
        patch.object(alerter_module.time, "monotonic", side_effect=[1000.0, 1100.0]) as mock_monotonic,
        patch.object(EmailBox, "send", side_effect=[None]) as mock_send,
    ):
        tested.alerted("emails", "SMTPDataError")
        tested.alerted("emails", "TimeoutError")
    exp_calls = [call(), call()]
    assert mock_monotonic.mock_calls == exp_calls
    exp_calls = [
        call(
            "ops@example.com",
            "[RAG My Docs] a background job failed",
            [
                "The background job 'emails' failed: SMTPDataError.",
                "The details are in the logs: docker logs rag_my_docs-app-<color>.",
            ],
        )
    ]
    assert mock_send.mock_calls == exp_calls


def test_alerted__send_failure_logged(caplog: pytest.LogCaptureFixture) -> None:
    tested = helper_instance()
    with (
        patch.object(Alerter, "_is_due", side_effect=[True]) as mock_is_due,
        patch.object(
            EmailBox, "send", side_effect=[AppException(502, "The mail relay refused the message.")]
        ) as mock_send,
        caplog.at_level(logging.WARNING, logger="rag_my_docs"),
    ):
        result = tested.alerted("emails", "SMTPDataError")
    assert result is None
    exp_messages = ["[ALERT] could not send the alert for emails"]
    assert [record.getMessage() for record in caplog.records] == exp_messages
    exp_calls = [call("emails")]
    assert mock_is_due.mock_calls == exp_calls
    exp_calls = [
        call(
            "ops@example.com",
            "[RAG My Docs] a background job failed",
            [
                "The background job 'emails' failed: SMTPDataError.",
                "The details are in the logs: docker logs rag_my_docs-app-<color>.",
            ],
        )
    ]
    assert mock_send.mock_calls == exp_calls


def test__is_due() -> None:
    """Once per job per hour: the first time, not 3599 s later, again at 3600 s; each job on its own clock."""
    tested = helper_instance()
    with patch.object(alerter_module.time, "monotonic", side_effect=[1000.0, 4599.0, 4600.0, 4601.0]) as mock_monotonic:
        result = [
            tested._is_due("emails"),
            tested._is_due("emails"),
            tested._is_due("emails"),
            tested._is_due("cleanup"),
        ]
    expected = [True, False, True, True]
    assert result == expected
    exp_last_sent = {"emails": 4600.0, "cleanup": 4601.0}
    assert tested._last_sent == exp_last_sent
    exp_calls = [call(), call(), call(), call()]
    assert mock_monotonic.mock_calls == exp_calls
