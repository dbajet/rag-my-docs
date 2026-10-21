from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from rag_my_docs.commands.evaluation_command import EvaluationCommand
from rag_my_docs.handlers.evaluation_run_request import EvaluationRunRequest
from rag_my_docs.handlers.session_guard import SessionGuard


class EvaluationRouter:
    """The test set: listed, and run one question at a time."""

    def __init__(self, guard: SessionGuard, evaluations: EvaluationCommand) -> None:
        self._guard = guard
        self._evaluations = evaluations

    def router(self) -> APIRouter:
        result = APIRouter(prefix="/api/evaluation")
        result.add_api_route("/cases", self.cases, methods=["GET"])
        result.add_api_route("/run", self.run, methods=["POST"])
        return result

    def cases(self, request: Request) -> dict[str, Any]:
        self._guard.user(request)
        return {"cases": [case.to_dict() for case in self._evaluations.cases()]}

    def run(self, request: Request, body: EvaluationRunRequest) -> dict[str, Any]:
        user = self._guard.user(request)
        return self._evaluations.evaluated(user, body.index, body.top_k).to_dict()
