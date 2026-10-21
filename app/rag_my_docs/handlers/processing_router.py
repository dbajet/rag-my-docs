from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from rag_my_docs.commands.processing_command import ProcessingCommand
from rag_my_docs.handlers.session_guard import SessionGuard


class ProcessingRouter:
    """What the pipeline made of a person's own document, for the inspection
    view, and the way to run it again."""

    def __init__(self, guard: SessionGuard, processing: ProcessingCommand) -> None:
        self._guard = guard
        self._processing = processing

    def router(self) -> APIRouter:
        result = APIRouter(prefix="/api/documents/{document_id}")
        result.add_api_route("/processing", self.processing, methods=["GET"])
        result.add_api_route("/processing", self.reprocessed, methods=["POST"])
        result.add_api_route("/extraction", self.extraction, methods=["GET"])
        return result

    def processing(self, request: Request, document_id: int) -> dict[str, Any]:
        """The status, the section tree and every chunk. `is_stale` says the
        chunks were made under other chunk sizes than the current ones."""
        found, sections, chunks = self._processing.processing(self._guard.user(request), document_id)
        strategy = self._processing.strategy()
        document = found.to_dict() | {
            "current_strategy": strategy,
            "is_stale": bool(found.chunking_version) and found.chunking_version != strategy,
        }
        return {
            "document": document,
            "sections": [section.to_dict() for section in sections],
            "chunks": [chunk.to_dict() for chunk in chunks],
        }

    def reprocessed(self, request: Request, document_id: int) -> dict[str, Any]:
        self._processing.reprocessed(self._guard.user(request), document_id)
        return {"ok": True}

    def extraction(self, request: Request, document_id: int) -> dict[str, Any]:
        blocks = self._processing.extraction(self._guard.user(request), document_id)
        return {"blocks": [block.to_dict() for block in blocks]}
