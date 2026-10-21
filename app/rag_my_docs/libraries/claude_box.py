from __future__ import annotations

import logging
import time
from typing import Any

import anthropic

from rag_my_docs.constants.constants import Constants
from rag_my_docs.constants.llm_constants import LlmConstants
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.connection_health import ConnectionHealth
from rag_my_docs.structures.llm_connection import LlmConnection


class ClaudeBox:
    """Claude, through Anthropic's SDK: one Messages request per call. Claude
    thinks before it answers, so its depth is set by `effort` rather than
    turned off.

    A client per call, with the connection's key, the step's timeout and no
    retry: a retried call would outlast the proxy. A declined request
    (`stop_reason: "refusal"`) is re-run on another model inside the same call
    for the models that offer it (`fallbacks: "default"`); one still declined
    is reported as such, never read as an answer."""

    def reply(
        self,
        connection: LlmConnection,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        effort: str,
        timeout: float,
    ) -> str:
        client = anthropic.Anthropic(api_key=connection.api_key, timeout=timeout, max_retries=0)
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": effort},
        }
        if model in LlmConstants.claude_fallback_models:
            request |= {"betas": [LlmConstants.claude_fallback_beta], "fallbacks": "default"}
        try:
            response = client.beta.messages.create(**request)
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as error:
            logging.getLogger(Constants.logger_name).warning("[LLM] %s from %s", type(error).__name__, model)
            raise AppException(503, LlmConstants.unavailable) from None
        if response.stop_reason == "refusal":
            raise AppException(422, LlmConstants.refused)
        return "".join(block.text for block in response.content if block.type == "text")

    def health(self, connection: LlmConnection, timeout: float) -> ConnectionHealth:
        """Green when the key is accepted and the model exists."""
        started = time.perf_counter()
        client = anthropic.Anthropic(api_key=connection.api_key, timeout=timeout, max_retries=0)
        detail = LlmConstants.health_ok
        is_ok = True
        try:
            client.models.retrieve(connection.model)
        except anthropic.NotFoundError:
            detail = f"Claude has no model {connection.model}."
            is_ok = False
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
            detail = LlmConstants.health_key_refused
            is_ok = False
        except anthropic.APIConnectionError:
            detail = LlmConstants.health_unreachable
            is_ok = False
        except anthropic.APIStatusError as error:
            detail = f"Claude answered HTTP {error.status_code}."
            is_ok = False
        return ConnectionHealth(
            connection_id=connection.connection_id,
            is_ok=is_ok,
            detail=detail,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def models(self, connection: LlmConnection, timeout: float) -> list[str]:
        """The models the key may use, newest first as Anthropic lists them,
        to choose from in Settings; refused in words a person can act on."""
        client = anthropic.Anthropic(api_key=connection.api_key, timeout=timeout, max_retries=0)
        try:
            # One page: far more than Anthropic lists, so no second request.
            return [model.id for model in client.models.list(limit=LlmConstants.claude_model_page).data]
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
            detail = LlmConstants.health_key_refused
        except anthropic.APIConnectionError:
            detail = LlmConstants.health_unreachable
        except anthropic.APIStatusError as error:
            detail = f"Claude answered HTTP {error.status_code}."
        raise AppException(502, detail)
