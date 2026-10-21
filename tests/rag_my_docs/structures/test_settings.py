from __future__ import annotations

import pytest

from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.settings import Settings
from tests.conftest import is_namedtuple


def helper_instance(environment: str) -> Settings:
    return Settings(
        environment=environment,
        database_url="postgresql://u:p@h:5432/d",
        app_version="2026-10-19.1",
        build_time="20261019-120000Z",
        app_commit="abc123",
        app_base_url="https://docs.example.com",
        data_key="the-data-key",
        smtp_host="smtp.example.com",
        smtp_port=587,
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


def helper_with_base_url(app_base_url: str) -> Settings:
    return helper_instance("dev")._replace(app_base_url=app_base_url)


def test_class() -> None:
    tested = Settings
    fields = {
        "environment": "str",
        "database_url": "str",
        "app_version": "str",
        "build_time": "str",
        "app_commit": "str",
        "app_base_url": "str",
        "data_key": "str",
        "smtp_host": "str",
        "smtp_port": "int",
        "smtp_user": "str",
        "smtp_password": "str",
        "email_from": "str",
        "bootstrap_admin_email": "str",
        "chunking": "ChunkingSettings",
        "embedding": "EmbeddingSettings",
    }
    assert is_namedtuple(tested, fields)


@pytest.mark.parametrize(
    "environment, expected",
    [
        pytest.param("production", True, id="production"),
        pytest.param("dev", False, id="dev"),
    ],
)
def test_is_production(environment: str, expected: bool) -> None:
    tested = helper_instance(environment)
    result = tested.is_production()
    assert result is expected


@pytest.mark.parametrize(
    "app_base_url, expected",
    [
        pytest.param("https://docs.example.com", "https://docs.example.com", id="plain"),
        pytest.param("http://localhost:8140", "http://localhost:8140", id="with-port"),
        pytest.param("https://docs.example.com/some/path?q=1", "https://docs.example.com", id="path-dropped"),
    ],
)
def test_origin(app_base_url: str, expected: str) -> None:
    tested = helper_with_base_url(app_base_url)
    result = tested.origin()
    assert result == expected


def test_to_dict() -> None:
    tested = helper_instance("dev")
    result = tested.to_dict()
    expected = {
        "environment": "dev",
        "database_url": "postgresql://u:p@h:5432/d",
        "app_version": "2026-10-19.1",
        "build_time": "20261019-120000Z",
        "app_commit": "abc123",
        "app_base_url": "https://docs.example.com",
        "data_key": "the-data-key",
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "smtp_user": "smtp-user",
        "smtp_password": "smtp-password",
        "email_from": "RAG My Docs <noreply@example.com>",
        "bootstrap_admin_email": "ada@example.com",
        "chunking": {"target_tokens": 200, "max_tokens": 350, "min_tokens": 75, "overlap_sentences": 1},
        "embedding": {
            "model": "BAAI/bge-small-en-v1.5",
            "query_prefix": "Represent this sentence for searching relevant passages: ",
            "batch_size": 32,
        },
    }
    assert result == expected


def test_from_dict() -> None:
    tested = Settings
    result = tested.from_dict(
        {
            "environment": "production",
            "database_url": "postgresql://u:p@h:5432/d",
            "app_version": "2026-10-19.1",
            "build_time": "20261019-120000Z",
            "app_commit": "abc123",
            "app_base_url": "https://docs.example.com",
            "data_key": "the-data-key",
            "smtp_host": "smtp.example.com",
            "smtp_port": "587",
            "smtp_user": "smtp-user",
            "smtp_password": "smtp-password",
            "email_from": "RAG My Docs <noreply@example.com>",
            "bootstrap_admin_email": "ada@example.com",
            "chunking": {"target_tokens": "200", "max_tokens": "350", "min_tokens": "75", "overlap_sentences": "1"},
            "embedding": {
                "model": "BAAI/bge-small-en-v1.5",
                "query_prefix": "Represent this sentence for searching relevant passages: ",
                "batch_size": "32",
            },
        }
    )
    expected = helper_instance("production")
    assert result == expected


def test_from_dict__a_missing_field_raises() -> None:
    tested = Settings
    with pytest.raises(KeyError):
        tested.from_dict({"environment": "dev"})
