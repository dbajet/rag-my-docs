from __future__ import annotations

import logging
import os
from unittest.mock import call, patch

import pytest

from rag_my_docs.libraries import settings_loader as settings_loader_module
from rag_my_docs.libraries.settings_loader import SettingsLoader
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.settings import Settings


def test___init__() -> None:
    environment = {"APP_ENV": "dev"}
    tested = SettingsLoader(environment)
    assert tested._environment is environment


def test_of_process() -> None:
    tested = SettingsLoader
    with patch.dict(os.environ, {"APP_ENV": "from-the-process"}):
        result = tested.of_process()
        assert result._read("APP_ENV") == "from-the-process"


def test_loaded() -> None:
    tested = SettingsLoader(
        {
            "APP_ENV": "production",
            "DATABASE_URL": " postgresql://u:secret@rag_my_docs-db:5432/rag_my_docs ",
            "APP_VERSION": "2026-10-11.2",
            "BUILD_TIME": "20261011-120000Z",
            "APP_COMMIT": "0123abcd",
            "APP_BASE_URL": "https://docs.example.com/",
            "DATA_ENCRYPTION_KEY": "a-real-data-key",
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "2587",
            "SMTP_USER": "smtp-user",
            "SMTP_PASSWORD": "smtp-password",
            "EMAIL_FROM": "RAG My Docs <noreply@example.com>",
            "BOOTSTRAP_ADMIN_EMAIL": " Ada@Example.COM ",
        }
    )
    result = tested.loaded()
    expected = Settings(
        environment="production",
        database_url="postgresql://u:secret@rag_my_docs-db:5432/rag_my_docs",
        app_version="2026-10-11.2",
        build_time="20261011-120000Z",
        app_commit="0123abcd",
        app_base_url="https://docs.example.com",
        data_key="a-real-data-key",
        smtp_host="smtp.example.com",
        smtp_port=2587,
        smtp_user="smtp-user",
        smtp_password="smtp-password",
        email_from="RAG My Docs <noreply@example.com>",
        bootstrap_admin_email="ada@example.com",
        chunking=ChunkingSettings(target_tokens=200, max_tokens=350, min_tokens=75, overlap_sentences=1),
        embedding=EmbeddingSettings(
            model="BAAI/bge-small-en-v1.5",
            query_prefix="Represent this sentence for searching relevant passages: ",
            batch_size=32,
        ),
    )
    assert result == expected


def test_loaded__defaults_for_a_laptop(caplog: pytest.LogCaptureFixture) -> None:
    tested = SettingsLoader({"DATABASE_URL": "postgresql://u:p@127.0.0.1:5540/rag_my_docs"})
    with (
        patch.object(settings_loader_module.secrets, "token_bytes", side_effect=[bytes(32)]) as mock_token_bytes,
        caplog.at_level(logging.WARNING, logger="rag_my_docs"),
    ):
        result = tested.loaded()
    expected = Settings(
        environment="dev",
        database_url="postgresql://u:p@127.0.0.1:5540/rag_my_docs",
        app_version="dev",
        build_time="dev",
        app_commit="dev",
        app_base_url="http://localhost:8140",
        data_key="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        smtp_host="",
        smtp_port=587,
        smtp_user="",
        smtp_password="",
        email_from="",
        bootstrap_admin_email="",
        chunking=ChunkingSettings(target_tokens=200, max_tokens=350, min_tokens=75, overlap_sentences=1),
        embedding=EmbeddingSettings(
            model="BAAI/bge-small-en-v1.5",
            query_prefix="Represent this sentence for searching relevant passages: ",
            batch_size=32,
        ),
    )
    assert result == expected
    exp_messages = ["[SETTINGS] DATA_ENCRYPTION_KEY is not set: generated for this process"]
    assert [record.getMessage() for record in caplog.records] == exp_messages
    exp_calls = [call(32)]
    assert mock_token_bytes.mock_calls == exp_calls


@pytest.mark.parametrize(
    "environment, expected",
    [
        pytest.param({}, "DATABASE_URL is not set", id="missing"),
        pytest.param({"DATABASE_URL": "   "}, "DATABASE_URL is not set", id="blank"),
    ],
)
def test__database_url__missing(environment: dict[str, str], expected: str) -> None:
    tested = SettingsLoader(environment)
    with pytest.raises(AppException) as exc_info:
        tested._database_url(False)
    assert exc_info.value.status_code == 500
    assert exc_info.value.message == expected


def test__database_url__placeholder_refused_in_production() -> None:
    tested = SettingsLoader({"DATABASE_URL": "postgresql://u:CHANGE_ME_token@h:5432/d"})
    with pytest.raises(AppException) as exc_info:
        tested._database_url(True)
    assert exc_info.value.status_code == 500
    assert exc_info.value.message == "DATABASE_URL still holds a placeholder"


def test__database_url__placeholder_warned_in_development(caplog: pytest.LogCaptureFixture) -> None:
    tested = SettingsLoader({"DATABASE_URL": "postgresql://u:CHANGE_ME_token@h:5432/d"})
    with caplog.at_level(logging.WARNING, logger="rag_my_docs"):
        result = tested._database_url(False)
    expected = "postgresql://u:CHANGE_ME_token@h:5432/d"
    assert result == expected
    exp_messages = ["[SETTINGS] DATABASE_URL still holds a placeholder"]
    assert [record.getMessage() for record in caplog.records] == exp_messages


@pytest.mark.parametrize(
    "environment, is_production, expected",
    [
        pytest.param({"APP_BASE_URL": "https://docs.example.com"}, True, "https://docs.example.com", id="said"),
        pytest.param(
            {"APP_BASE_URL": " https://docs.example.com// "}, True, "https://docs.example.com", id="trailing-slash"
        ),
        pytest.param({}, False, "http://localhost:8140", id="laptop-default"),
        pytest.param({"APP_BASE_URL": "/"}, False, "http://localhost:8140", id="only-a-slash"),
        pytest.param({"APP_BASE_URL": "http://127.0.0.1:8140"}, False, "http://127.0.0.1:8140", id="laptop-http"),
    ],
)
def test__app_base_url(environment: dict[str, str], is_production: bool, expected: str) -> None:
    tested = SettingsLoader(environment)
    result = tested._app_base_url(is_production)
    assert result == expected


def test__app_base_url__refused_in_production() -> None:
    tested = SettingsLoader({})
    with pytest.raises(AppException) as exc_info:
        tested._app_base_url(True)
    assert exc_info.value.status_code == 500
    assert exc_info.value.message == "APP_BASE_URL is not set"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("http://docs.example.com", id="plain-http"),
        pytest.param("https://", id="no-host"),
        pytest.param("docs.example.com", id="no-scheme"),
    ],
)
def test__app_base_url__production_wants_https_with_a_host(value: str) -> None:
    tested = SettingsLoader({"APP_BASE_URL": value})
    with pytest.raises(AppException) as exc_info:
        tested._app_base_url(True)
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (500, "APP_BASE_URL must be an https:// address with a host")
    assert result == expected


@pytest.mark.parametrize(
    "is_production",
    [
        pytest.param(True, id="production"),
        pytest.param(False, id="dev"),
    ],
)
def test__data_key(is_production: bool) -> None:
    tested = SettingsLoader({"DATA_ENCRYPTION_KEY": " a-real-data-key "})
    result = tested._data_key(is_production)
    expected = "a-real-data-key"
    assert result == expected


@pytest.mark.parametrize(
    "environment",
    [
        pytest.param({}, id="missing"),
        pytest.param({"DATA_ENCRYPTION_KEY": "CHANGE_ME_fernet_key"}, id="placeholder"),
    ],
)
def test__data_key__refused_in_production(environment: dict[str, str]) -> None:
    tested = SettingsLoader(environment)
    with pytest.raises(AppException) as exc_info:
        tested._data_key(True)
    assert exc_info.value.status_code == 500
    assert exc_info.value.message == "DATA_ENCRYPTION_KEY is not set, or is still a placeholder"


@pytest.mark.parametrize(
    "environment",
    [
        pytest.param({}, id="missing"),
        pytest.param({"DATA_ENCRYPTION_KEY": "CHANGE_ME_fernet_key"}, id="placeholder"),
    ],
)
def test__data_key__generated_in_development(environment: dict[str, str], caplog: pytest.LogCaptureFixture) -> None:
    tested = SettingsLoader(environment)
    with (
        patch.object(settings_loader_module.secrets, "token_bytes", side_effect=[bytes(range(32))]) as mock_token_bytes,
        caplog.at_level(logging.WARNING, logger="rag_my_docs"),
    ):
        result = tested._data_key(False)
    expected = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="
    assert result == expected
    exp_messages = ["[SETTINGS] DATA_ENCRYPTION_KEY is not set: generated for this process"]
    assert [record.getMessage() for record in caplog.records] == exp_messages
    exp_calls = [call(32)]
    assert mock_token_bytes.mock_calls == exp_calls


@pytest.mark.parametrize(
    "environment, is_production, expected",
    [
        pytest.param({"SMTP_HOST": " smtp.example.com "}, True, "smtp.example.com", id="production-relay"),
        pytest.param({"SMTP_HOST": "smtp.example.com"}, False, "smtp.example.com", id="dev-relay"),
        pytest.param({}, False, "", id="dev-no-relay"),
    ],
)
def test__smtp_host(environment: dict[str, str], is_production: bool, expected: str) -> None:
    tested = SettingsLoader(environment)
    result = tested._smtp_host(is_production)
    assert result == expected


def test__smtp_host__refused_in_production() -> None:
    tested = SettingsLoader({})
    with pytest.raises(AppException) as exc_info:
        tested._smtp_host(True)
    assert exc_info.value.status_code == 500
    assert exc_info.value.message == "SMTP_HOST is not set: in production a link must reach an inbox"


@pytest.mark.parametrize(
    "environment, expected",
    [
        pytest.param({}, 587, id="default"),
        pytest.param({"SMTP_PORT": " "}, 587, id="blank"),
        pytest.param({"SMTP_PORT": "2465"}, 2465, id="said"),
    ],
)
def test__smtp_port(environment: dict[str, str], expected: int) -> None:
    tested = SettingsLoader(environment)
    result = tested._smtp_port()
    assert result == expected


def test__smtp_port__not_a_number() -> None:
    tested = SettingsLoader({"SMTP_PORT": "five-eight-seven"})
    with pytest.raises(AppException) as exc_info:
        tested._smtp_port()
    assert exc_info.value.status_code == 500
    assert exc_info.value.message == "SMTP_PORT is not a port: 'five-eight-seven'"


@pytest.mark.parametrize(
    "environment, expected",
    [
        pytest.param(
            {}, ChunkingSettings(target_tokens=200, max_tokens=350, min_tokens=75, overlap_sentences=1), id="defaults"
        ),
        pytest.param(
            {
                "CHUNK_TARGET_TOKENS": "120",
                "CHUNK_MAX_TOKENS": " 180 ",
                "CHUNK_MIN_TOKENS": "40",
                "CHUNK_OVERLAP_SENTENCES": "0",
            },
            ChunkingSettings(target_tokens=120, max_tokens=180, min_tokens=40, overlap_sentences=0),
            id="said",
        ),
        pytest.param(
            {"CHUNK_TARGET_TOKENS": "5", "CHUNK_MAX_TOKENS": "5", "CHUNK_MIN_TOKENS": "5"},
            ChunkingSettings(target_tokens=5, max_tokens=5, min_tokens=5, overlap_sentences=1),
            id="all-equal",
        ),
    ],
)
def test__chunking(environment: dict[str, str], expected: ChunkingSettings) -> None:
    tested = SettingsLoader(environment)
    result = tested._chunking()
    assert result == expected


@pytest.mark.parametrize(
    "environment",
    [
        pytest.param({"CHUNK_TARGET_TOKENS": "400"}, id="target-over-max"),
        pytest.param({"CHUNK_MIN_TOKENS": "250"}, id="min-over-target"),
        pytest.param({"CHUNK_MIN_TOKENS": "0"}, id="min-zero"),
        pytest.param({"CHUNK_OVERLAP_SENTENCES": "-1"}, id="negative-overlap"),
    ],
)
def test__chunking__sizes_that_cannot_work_together(environment: dict[str, str]) -> None:
    tested = SettingsLoader(environment)
    with pytest.raises(AppException) as exc_info:
        tested._chunking()
    assert exc_info.value.status_code == 500
    expected = (
        "Chunk sizes must satisfy 1 <= CHUNK_MIN_TOKENS <= CHUNK_TARGET_TOKENS <= CHUNK_MAX_TOKENS, "
        "and CHUNK_OVERLAP_SENTENCES >= 0"
    )
    assert exc_info.value.message == expected


@pytest.mark.parametrize(
    "environment, expected",
    [
        pytest.param(
            {},
            EmbeddingSettings(
                model="BAAI/bge-small-en-v1.5",
                query_prefix="Represent this sentence for searching relevant passages: ",
                batch_size=32,
            ),
            id="defaults",
        ),
        pytest.param(
            {
                "EMBEDDING_MODEL": " intfloat/e5-small-v2 ",
                "EMBEDDING_QUERY_PREFIX": "query: ",
                "EMBEDDING_BATCH_SIZE": "8",
            },
            EmbeddingSettings(model="intfloat/e5-small-v2", query_prefix="query: ", batch_size=8),
            id="said",
        ),
        # A model without an instruction: an empty prefix is kept, not replaced by the default.
        pytest.param(
            {"EMBEDDING_MODEL": "", "EMBEDDING_QUERY_PREFIX": "", "EMBEDDING_BATCH_SIZE": "1"},
            EmbeddingSettings(model="BAAI/bge-small-en-v1.5", query_prefix="", batch_size=1),
            id="empty-prefix",
        ),
    ],
)
def test__embedding(environment: dict[str, str], expected: EmbeddingSettings) -> None:
    tested = SettingsLoader(environment)
    result = tested._embedding()
    assert result == expected


def test__embedding__a_batch_below_one_is_refused() -> None:
    tested = SettingsLoader({"EMBEDDING_BATCH_SIZE": "0"})
    with pytest.raises(AppException) as exc_info:
        tested._embedding()
    result = (exc_info.value.status_code, exc_info.value.message)
    expected = (500, "EMBEDDING_BATCH_SIZE must be at least 1")
    assert result == expected


@pytest.mark.parametrize(
    "environment, expected",
    [
        pytest.param({}, 42, id="default"),
        pytest.param({"SOME_NUMBER": "  "}, 42, id="blank"),
        pytest.param({"SOME_NUMBER": "7"}, 7, id="said"),
        pytest.param({"SOME_NUMBER": "-3"}, -3, id="negative"),
    ],
)
def test__number(environment: dict[str, str], expected: int) -> None:
    tested = SettingsLoader(environment)
    result = tested._number("SOME_NUMBER", 42)
    assert result == expected


def test__number__not_a_whole_number() -> None:
    tested = SettingsLoader({"CHUNK_MAX_TOKENS": "3.5"})
    with pytest.raises(AppException) as exc_info:
        tested._number("CHUNK_MAX_TOKENS", 350)
    assert exc_info.value.status_code == 500
    assert exc_info.value.message == "CHUNK_MAX_TOKENS is not a whole number: '3.5'"


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param("CHANGE_ME_token_urlsafe_24", True, id="placeholder-upper"),
        pytest.param("postgresql://u:change-me@h/d", True, id="dashed-inside"),
        pytest.param("placeholder", True, id="word"),
        pytest.param("postgresql://u:s3cret@h/d", False, id="real"),
        pytest.param("", False, id="empty"),
    ],
)
def test_has_placeholder(value: str, expected: bool) -> None:
    tested = SettingsLoader
    result = tested.has_placeholder(value)
    assert result is expected


def test__warned(caplog: pytest.LogCaptureFixture) -> None:
    tested = SettingsLoader
    with caplog.at_level(logging.WARNING, logger="rag_my_docs"):
        tested._warned("SOME_NAME", "is odd")
    exp_messages = ["[SETTINGS] SOME_NAME is odd"]
    assert [record.getMessage() for record in caplog.records] == exp_messages


def test__read() -> None:
    tested = SettingsLoader({"A": "  value \n"})
    result = (tested._read("A"), tested._read("B"))
    expected = ("value", "")
    assert result == expected
