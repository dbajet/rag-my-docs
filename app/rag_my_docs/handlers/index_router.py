from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from rag_my_docs.commands.index_command import IndexCommand
from rag_my_docs.handlers.index_request import IndexRequest
from rag_my_docs.handlers.session_guard import SessionGuard


class IndexRouter:
    """Asking for a person's documents to be indexed, one or all."""

    def __init__(self, guard: SessionGuard, indexing: IndexCommand) -> None:
        self._guard = guard
        self._indexing = indexing

    def router(self) -> APIRouter:
        result = APIRouter(prefix="/api")
        result.add_api_route("/documents/{document_id}/index", self.reindexed, methods=["POST"])
        result.add_api_route("/index/all", self.indexed_all, methods=["POST"])
        return result

    def reindexed(self, request: Request, document_id: int, body: IndexRequest) -> dict[str, Any]:
        self._indexing.reindexed(self._guard.user(request), document_id, body.rebuild)
        return {"ok": True}

    def indexed_all(self, request: Request) -> dict[str, Any]:
        return {"queued": self._indexing.indexed_all(self._guard.user(request))}
