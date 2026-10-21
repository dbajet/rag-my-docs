from __future__ import annotations

from typing import Any, NamedTuple
from urllib.parse import urlsplit

from rag_my_docs.constants.constants import Constants
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.embedding_settings import EmbeddingSettings


class Settings(NamedTuple):
    """Everything this process was told by its environment, read once by
    `SettingsLoader` and injected everywhere else."""

    environment: str
    database_url: str
    # What is running, stamped by the deploy script because only it knows.
    app_version: str
    build_time: str
    app_commit: str
    # Links come from this, never from a request's Host.
    app_base_url: str
    # Seals the documents, their names and the emails, and keys the blind index.
    # Changing it makes every sealed value unreadable, so a deployment keeps the
    # one it started with.
    data_key: str
    # Without a host, a sign-in link is written to the log: a laptop's choice, refused in production.
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    email_from: str
    # The first admin, created at startup while no admin exists, then ignored:
    # a variable that could re-create an admin would be a back door.
    bootstrap_admin_email: str
    # How documents are cut into chunks; changing it makes processed documents stale.
    chunking: ChunkingSettings
    # The local embedding model; the image carries its weights (Dockerfile).
    embedding: EmbeddingSettings

    def is_production(self) -> bool:
        return self.environment == Constants.environment_production

    def origin(self) -> str:
        """The scheme, host and port the browser shows: what a request's
        `Origin` must equal."""
        parts = urlsplit(self.app_base_url)
        return f"{parts.scheme}://{parts.netloc}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "database_url": self.database_url,
            "app_version": self.app_version,
            "build_time": self.build_time,
            "app_commit": self.app_commit,
            "app_base_url": self.app_base_url,
            "data_key": self.data_key,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "smtp_user": self.smtp_user,
            "smtp_password": self.smtp_password,
            "email_from": self.email_from,
            "bootstrap_admin_email": self.bootstrap_admin_email,
            "chunking": self.chunking.to_dict(),
            "embedding": self.embedding.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        return cls(
            environment=str(data["environment"]),
            database_url=str(data["database_url"]),
            app_version=str(data["app_version"]),
            build_time=str(data["build_time"]),
            app_commit=str(data["app_commit"]),
            app_base_url=str(data["app_base_url"]),
            data_key=str(data["data_key"]),
            smtp_host=str(data["smtp_host"]),
            smtp_port=int(data["smtp_port"]),
            smtp_user=str(data["smtp_user"]),
            smtp_password=str(data["smtp_password"]),
            email_from=str(data["email_from"]),
            bootstrap_admin_email=str(data["bootstrap_admin_email"]),
            chunking=ChunkingSettings.from_dict(data["chunking"]),
            embedding=EmbeddingSettings.from_dict(data["embedding"]),
        )
