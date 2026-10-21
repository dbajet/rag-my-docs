from __future__ import annotations

import base64
import logging
import os
import secrets
from collections.abc import Mapping
from urllib.parse import urlsplit

from rag_my_docs.constants.constants import Constants
from rag_my_docs.constants.embedding_constants import EmbeddingConstants
from rag_my_docs.constants.processing_constants import ProcessingConstants
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.settings import Settings


class SettingsLoader:
    """The one place that reads the environment.

    It fails closed in production and warns in development, here rather than
    only in the deploy script: a `docker compose up` run by hand has no deploy
    script in front of it, and would otherwise boot on a placeholder.
    """

    def __init__(self, environment: Mapping[str, str]) -> None:
        self._environment = environment

    @classmethod
    def of_process(cls) -> SettingsLoader:
        """This process's own environment: the one mention of `os.environ`
        in the application."""
        return cls(os.environ)

    def loaded(self) -> Settings:
        environment = self._read(Constants.env_environment) or Constants.environment_development
        is_production = environment == Constants.environment_production
        result = Settings(
            environment=environment,
            database_url=self._database_url(is_production),
            app_version=self._read(Constants.env_app_version) or Constants.version_unstamped,
            build_time=self._read(Constants.env_build_time) or Constants.version_unstamped,
            app_commit=self._read(Constants.env_app_commit) or Constants.version_unstamped,
            app_base_url=self._app_base_url(is_production),
            data_key=self._data_key(is_production),
            smtp_host=self._smtp_host(is_production),
            smtp_port=self._smtp_port(),
            smtp_user=self._read(Constants.env_smtp_user),
            smtp_password=self._read(Constants.env_smtp_password),
            email_from=self._read(Constants.env_email_from),
            bootstrap_admin_email=self._read(Constants.env_bootstrap_admin_email).lower(),
            chunking=self._chunking(),
            embedding=self._embedding(),
        )
        return result

    def _database_url(self, is_production: bool) -> str:
        """No default in any environment: a process with nowhere to keep its
        rows has nothing to offer. The URL carries the database password, so
        a placeholder in it is refused in production like any other secret."""
        result = self._read(Constants.env_database_url)
        if not result:
            raise AppException(500, f"{Constants.env_database_url} is not set")
        if self.has_placeholder(result):
            if is_production:
                raise AppException(500, f"{Constants.env_database_url} still holds a placeholder")
            self._warned(Constants.env_database_url, "still holds a placeholder")
        return result

    def _app_base_url(self, is_production: bool) -> str:
        """Every link is made from this: in production it must be said, never
        guessed from a request."""
        result = self._read(Constants.env_app_base_url).rstrip("/")
        if not result:
            if is_production:
                raise AppException(500, f"{Constants.env_app_base_url} is not set")
            return Constants.default_app_base_url
        # Links need a host, and cookies marked Secure need TLS.
        parts = urlsplit(result)
        if is_production and (parts.scheme != "https" or not parts.hostname):
            raise AppException(500, f"{Constants.env_app_base_url} must be an https:// address with a host")
        return result

    def _data_key(self, is_production: bool) -> str:
        """The sealing key, or a fresh one per process on a laptop: a restart
        then forgets what it sealed, which a laptop can afford. Production
        refuses: its sealed documents must open after every restart."""
        result = self._read(Constants.env_data_key)
        if result and not self.has_placeholder(result):
            return result
        if is_production:
            raise AppException(500, f"{Constants.env_data_key} is not set, or is still a placeholder")
        self._warned(Constants.env_data_key, "is not set: generated for this process")
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(Constants.text_encoding)

    def _smtp_host(self, is_production: bool) -> str:
        """The relay, or nothing, which writes sign-in links to the log. A
        link in a log is a login for whoever reads the log, so production
        refuses to start without a relay."""
        result = self._read(Constants.env_smtp_host)
        if not result and is_production:
            raise AppException(500, f"{Constants.env_smtp_host} is not set: in production a link must reach an inbox")
        return result

    def _smtp_port(self) -> int:
        """A port that is not a number is a mistake worth stopping on."""
        raw = self._read(Constants.env_smtp_port)
        if not raw:
            return Constants.smtp_default_port
        try:
            return int(raw)
        except ValueError as error:
            raise AppException(500, f"{Constants.env_smtp_port} is not a port: {raw!r}") from error

    def _chunking(self) -> ChunkingSettings:
        """Sizes that cannot work together stop the start: a chunk could never
        be both under its maximum and over its target."""
        result = ChunkingSettings(
            target_tokens=self._number(
                ProcessingConstants.env_target_tokens, ProcessingConstants.default_target_tokens
            ),
            max_tokens=self._number(ProcessingConstants.env_max_tokens, ProcessingConstants.default_max_tokens),
            min_tokens=self._number(ProcessingConstants.env_min_tokens, ProcessingConstants.default_min_tokens),
            overlap_sentences=self._number(
                ProcessingConstants.env_overlap_sentences, ProcessingConstants.default_overlap_sentences
            ),
        )
        if not 1 <= result.min_tokens <= result.target_tokens <= result.max_tokens or result.overlap_sentences < 0:
            raise AppException(
                500,
                f"Chunk sizes must satisfy 1 <= {ProcessingConstants.env_min_tokens} <= "
                f"{ProcessingConstants.env_target_tokens} <= {ProcessingConstants.env_max_tokens}, "
                f"and {ProcessingConstants.env_overlap_sentences} >= 0",
            )
        return result

    def _embedding(self) -> EmbeddingSettings:
        """The model and its query instruction; a batch of fewer than one text is a mistake."""
        result = EmbeddingSettings(
            model=self._read(EmbeddingConstants.env_model) or EmbeddingConstants.default_model,
            query_prefix=self._environment.get(
                EmbeddingConstants.env_query_prefix, EmbeddingConstants.default_query_prefix
            ),
            batch_size=self._number(EmbeddingConstants.env_batch_size, EmbeddingConstants.default_batch_size),
        )
        if result.batch_size < 1:
            raise AppException(500, f"{EmbeddingConstants.env_batch_size} must be at least 1")
        return result

    def _number(self, name: str, default: int) -> int:
        raw = self._read(name)
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError as error:
            raise AppException(500, f"{name} is not a whole number: {raw!r}") from error

    @classmethod
    def has_placeholder(cls, value: str) -> bool:
        lowered = value.lower()
        return any(marker in lowered for marker in Constants.placeholder_markers)

    @classmethod
    def _warned(cls, name: str, problem: str) -> None:
        logging.getLogger(Constants.logger_name).warning("[SETTINGS] %s %s", name, problem)

    def _read(self, name: str) -> str:
        return self._environment.get(name, "").strip()
