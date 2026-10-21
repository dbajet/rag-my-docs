from __future__ import annotations

import logging

import pytest

from rag_my_docs.libraries.email_box import EmailBox
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.settings import Settings
from tests.support.fake_smtp import FakeSMTP


def helper_settings(smtp_host: str, smtp_user: str) -> Settings:
    return Settings(
        environment="production",
        database_url="postgresql://u:p@h/d",
        app_version="2026-10-11.1",
        build_time="20261011-120000Z",
        app_commit="abc123",
        app_base_url="https://docs.example.com",
        data_key="the-data-key",
        smtp_host=smtp_host,
        smtp_port=587,
        smtp_user=smtp_user,
        smtp_password="smtp-password",
        email_from="RAG My Docs <noreply@example.com>",
        bootstrap_admin_email="",
        chunking=ChunkingSettings(target_tokens=200, max_tokens=350, min_tokens=75, overlap_sentences=1),
        embedding=EmbeddingSettings(
            model="BAAI/bge-small-en-v1.5",
            query_prefix="Represent this sentence for searching relevant passages: ",
            batch_size=32,
        ),
    )


def helper_instance(smtp_user: str = "smtp-user") -> EmailBox:
    return EmailBox(helper_settings("smtp.example.com", smtp_user))


def test___init__() -> None:
    tested = helper_instance()
    result = (tested._host, tested._port, tested._user, tested._password, tested._sender)
    expected = (
        "smtp.example.com",
        587,
        "smtp-user",
        "smtp-password",
        "RAG My Docs <noreply@example.com>",
    )
    assert result == expected


def test_send(caplog: pytest.LogCaptureFixture) -> None:
    tested = helper_instance()
    fake = FakeSMTP()
    with fake.installed(), caplog.at_level(logging.WARNING, logger="rag_my_docs"):
        result = tested.send("ada@example.com", "Your link", ["Hello Ada,", "https://x/auth/link?token=t"])
    assert result is None
    expected = [
        ("connect", "smtp.example.com", 587, 15.0),
        ("starttls", "CERT_REQUIRED", True),
        ("login", "smtp-user", "smtp-password"),
        (
            "send_message",
            "RAG My Docs <noreply@example.com>",
            "ada@example.com",
            "Your link",
            "Hello Ada,\nhttps://x/auth/link?token=t\n",
        ),
        ("quit",),
    ]
    assert fake.calls == expected
    exp_messages: list[str] = []
    assert [record.getMessage() for record in caplog.records] == exp_messages


def test_send__no_login_without_a_user() -> None:
    tested = helper_instance(smtp_user="")
    fake = FakeSMTP()
    with fake.installed():
        tested.send("ada@example.com", "Your link", ["Hello Ada,"])
    expected = [
        ("connect", "smtp.example.com", 587, 15.0),
        ("starttls", "CERT_REQUIRED", True),
        ("send_message", "RAG My Docs <noreply@example.com>", "ada@example.com", "Your link", "Hello Ada,\n"),
        ("quit",),
    ]
    assert fake.calls == expected


def test_send__no_relay_writes_the_log(caplog: pytest.LogCaptureFixture) -> None:
    tested = EmailBox(helper_settings("", "smtp-user"))
    fake = FakeSMTP()
    with fake.installed(), caplog.at_level(logging.WARNING, logger="rag_my_docs"):
        result = tested.send("ada@example.com", "Your link", ["Hello Ada,", "https://x/auth/link?token=t"])
    assert result is None
    expected: list[tuple[object, ...]] = []
    assert fake.calls == expected
    exp_messages = ["[EMAIL] no relay: Hello Ada, | https://x/auth/link?token=t"]
    assert [record.getMessage() for record in caplog.records] == exp_messages


@pytest.mark.parametrize(
    "failing_step, expected, exp_messages",
    [
        pytest.param(
            "connect",
            [("connect", "smtp.example.com", 587, 15.0)],
            ["[EMAIL] relay refused: SMTPConnectError"],
            id="connect",
        ),
        pytest.param(
            "starttls",
            [
                ("connect", "smtp.example.com", 587, 15.0),
                ("starttls", "CERT_REQUIRED", True),
                ("quit",),
            ],
            ["[EMAIL] relay refused: SMTPNotSupportedError"],
            id="starttls",
        ),
        pytest.param(
            "login",
            [
                ("connect", "smtp.example.com", 587, 15.0),
                ("starttls", "CERT_REQUIRED", True),
                ("login", "smtp-user", "smtp-password"),
                ("quit",),
            ],
            ["[EMAIL] relay refused: SMTPAuthenticationError"],
            id="login",
        ),
        pytest.param(
            "send_message",
            [
                ("connect", "smtp.example.com", 587, 15.0),
                ("starttls", "CERT_REQUIRED", True),
                ("login", "smtp-user", "smtp-password"),
                ("send_message", "RAG My Docs <noreply@example.com>", "ada@example.com", "Your link", "Hello Ada,\n"),
                ("quit",),
            ],
            ["[EMAIL] relay refused: SMTPDataError"],
            id="send-message",
        ),
    ],
)
def test_send__relay_refused(
    failing_step: str,
    expected: list[tuple[object, ...]],
    exp_messages: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    tested = helper_instance()
    fake = FakeSMTP(failing_step=failing_step)
    with (
        fake.installed(),
        caplog.at_level(logging.WARNING, logger="rag_my_docs"),
        pytest.raises(AppException) as exc_info,
    ):
        tested.send("ada@example.com", "Your link", ["Hello Ada,"])
    assert exc_info.value.status_code == 502
    assert exc_info.value.message == "The mail relay refused the message."
    assert fake.calls == expected
    assert [record.getMessage() for record in caplog.records] == exp_messages


def test_send__relay_unreachable(caplog: pytest.LogCaptureFixture) -> None:
    tested = helper_instance()
    fake = FakeSMTP(unreachable=True)
    with (
        fake.installed(),
        caplog.at_level(logging.WARNING, logger="rag_my_docs"),
        pytest.raises(AppException) as exc_info,
    ):
        tested.send("ada@example.com", "Your link", ["Hello Ada,"])
    assert exc_info.value.status_code == 502
    assert exc_info.value.message == "The mail relay could not be reached."
    expected = [("connect", "smtp.example.com", 587, 15.0)]
    assert fake.calls == expected
    exp_messages = ["[EMAIL] relay unreachable: ConnectionRefusedError"]
    assert [record.getMessage() for record in caplog.records] == exp_messages
