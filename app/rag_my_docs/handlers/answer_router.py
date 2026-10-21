from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from rag_my_docs.commands.answer_command import AnswerCommand
from rag_my_docs.handlers.answer_request import AnswerRequest
from rag_my_docs.handlers.session_guard import SessionGuard


class AnswerRouter:
    """Questions answered from the publications: retrieval, then the model."""

    def __init__(self, guard: SessionGuard, answers: AnswerCommand) -> None:
        self._guard = guard
        self._answers = answers

    def router(self) -> APIRouter:
        result = APIRouter(prefix="/api")
        result.add_api_route("/answers", self.answered, methods=["POST"])
        return result

    def answered(self, request: Request, body: AnswerRequest) -> dict[str, Any]:
        return self._answers.answered(self._guard.user(request), body.question, body.top_k).to_dict()
