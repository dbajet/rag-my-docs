from __future__ import annotations

from rag_my_docs.constants.llm_constants import LlmConstants
from rag_my_docs.constants.llm_kind import LlmKind
from rag_my_docs.libraries.claude_box import ClaudeBox
from rag_my_docs.libraries.llm_connection_store import LlmConnectionStore
from rag_my_docs.libraries.openai_chat_box import OpenAiChatBox
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.connection_health import ConnectionHealth
from rag_my_docs.structures.llm_connection import LlmConnection


class LlmGateway:
    """The answer to the connection chosen in Settings, through the box of
    that connection's kind, local or cloud. Read at every call, so a change
    in Settings applies to the next question."""

    def __init__(self, store: LlmConnectionStore, openai_box: OpenAiChatBox, claude_box: ClaudeBox) -> None:
        self._store = store
        self._openai_box = openai_box
        self._claude_box = claude_box

    def is_configured(self) -> bool:
        return self._store.routing().answers is not None

    def answer_reply(self, system: str, user: str) -> tuple[str, str]:
        """The answer, and the model that wrote it."""
        connection = self.connection_for(self._store.routing().answers)
        if connection.kind is LlmKind.CLAUDE:
            text = self._claude_box.reply(
                connection,
                connection.model,
                system,
                user,
                LlmConstants.claude_answer_max_tokens,
                LlmConstants.claude_answer_effort,
                LlmConstants.answer_timeout_seconds,
            )
        else:
            text = self._openai_box.reply(
                connection,
                connection.model,
                system,
                user,
                LlmConstants.answer_max_tokens,
                LlmConstants.answer_timeout_seconds,
            )
        return text, connection.model

    def health(self, connection: LlmConnection) -> ConnectionHealth:
        if connection.kind is LlmKind.CLAUDE:
            return self._claude_box.health(connection, LlmConstants.health_timeout_seconds)
        return self._openai_box.health(connection, LlmConstants.health_timeout_seconds)

    def models(self, connection: LlmConnection) -> list[str]:
        if connection.kind is LlmKind.CLAUDE:
            return self._claude_box.models(connection, LlmConstants.health_timeout_seconds)
        return self._openai_box.models(connection, LlmConstants.health_timeout_seconds)

    def connection_for(self, connection_id: int | None) -> LlmConnection:
        """The answering connection; refused while there is none."""
        found = None if connection_id is None else self._store.connection(connection_id)
        if found is None:
            raise AppException(503, LlmConstants.not_configured)
        return found
