from __future__ import annotations

from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import Response

from rag_my_docs.commands.document_command import DocumentCommand
from rag_my_docs.constants.constants import Constants
from rag_my_docs.constants.document_constants import DocumentConstants
from rag_my_docs.handlers.session_guard import SessionGuard
from rag_my_docs.structures.document_file import DocumentFile


class DocumentRouter:
    """The corpus: its publications listed, and each one's PDF."""

    def __init__(self, guard: SessionGuard, documents: DocumentCommand) -> None:
        self._guard = guard
        self._documents = documents

    def router(self) -> APIRouter:
        result = APIRouter(prefix="/api/documents")
        result.add_api_route("", self.documents, methods=["GET"])
        result.add_api_route("/{document_id}/content", self.downloaded, methods=["GET"])
        return result

    def documents(self, request: Request) -> dict[str, Any]:
        user = self._guard.user(request)
        return {"documents": [document.to_dict() for document in self._documents.documents(user)]}

    def downloaded(self, request: Request, document_id: int) -> Response:
        document = self._documents.downloaded(self._guard.user(request), document_id)
        return Response(
            document.content,
            media_type=document.media_type.value,
            headers={
                "Content-Disposition": self.disposition(document),
                Constants.header_cache_control: DocumentConstants.download_cache_control,
            },
        )

    @classmethod
    def disposition(cls, document: DocumentFile) -> str:
        """Always an attachment, never shown in the page's origin: a stored
        file is a download's bytes, and an HTML-looking one must not run here.
        RFC 6266: an ASCII fallback for old clients, the UTF-8 name for the rest."""
        fallback = "".join(
            character if character.isascii() and character.isprintable() and character not in '"\\' else "_"
            for character in document.name
        )
        return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(document.name, safe='')}"
