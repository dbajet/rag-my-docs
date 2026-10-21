from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from rag_my_docs.constants.constants import Constants
from rag_my_docs.constants.llm_constants import LlmConstants
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.connection_health import ConnectionHealth
from rag_my_docs.structures.llm_connection import LlmConnection


class OpenAiChatBox:
    """A chat-completions endpoint in the OpenAI format, which vLLM (a local
    machine) and many hosted providers speak: one POST per call, a timeout on
    each, and the reply's text. Hand-rolled with urllib: one request shape
    needs no SDK.

    What leaves the server: the question and the passages retrieved for it. Nothing is logged but the failure's kind and the model."""

    def reply(
        self,
        connection: LlmConnection,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        timeout: float,
    ) -> str:
        request = urllib.request.Request(
            f"{connection.base_url}{LlmConstants.chat_path}",
            data=json.dumps(self.body(connection, model, system, user, max_tokens)).encode(Constants.text_encoding),
            headers=self.headers(connection) | {"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read())
            return str(data["choices"][0]["message"]["content"] or "")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as error:
            # The provider's detail stays in the log; the person reads that it failed.
            logging.getLogger(Constants.logger_name).warning("[LLM] %s from %s", type(error).__name__, model)
            raise AppException(503, LlmConstants.unavailable) from None

    def health(self, connection: LlmConnection, timeout: float) -> ConnectionHealth:
        """Green when the endpoint lists the connection's model."""
        started = time.perf_counter()
        request = urllib.request.Request(
            f"{connection.base_url}{LlmConstants.models_path}", headers=self.headers(connection), method="GET"
        )
        is_ok = False
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                listed = {str(model["id"]) for model in json.loads(response.read())["data"]}
            is_ok = connection.model in listed
            detail = LlmConstants.health_ok if is_ok else f"The endpoint does not serve {connection.model}."
        except urllib.error.HTTPError as error:
            refused = error.code in (401, 403)
            detail = LlmConstants.health_key_refused if refused else f"The endpoint answered HTTP {error.code}."
        except (urllib.error.URLError, TimeoutError, OSError):
            detail = LlmConstants.health_unreachable
        except (ValueError, KeyError, TypeError):
            detail = LlmConstants.not_a_model_list
        return ConnectionHealth(
            connection_id=connection.connection_id,
            is_ok=is_ok,
            detail=detail,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def models(self, connection: LlmConnection, timeout: float) -> list[str]:
        """The models the endpoint serves, by name, to choose from in
        Settings; refused in words a person can act on."""
        request = urllib.request.Request(
            f"{connection.base_url}{LlmConstants.models_path}", headers=self.headers(connection), method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return sorted({str(model["id"]) for model in json.loads(response.read())["data"]})
        except urllib.error.HTTPError as error:
            refused = error.code in (401, 403)
            detail = LlmConstants.health_key_refused if refused else f"The endpoint answered HTTP {error.code}."
        except (urllib.error.URLError, TimeoutError, OSError):
            detail = LlmConstants.health_unreachable
        except (ValueError, KeyError, TypeError):
            detail = LlmConstants.not_a_model_list
        raise AppException(502, detail)

    @classmethod
    def headers(cls, connection: LlmConnection) -> dict[str, str]:
        """A local endpoint may need no key: then no Authorization is sent."""
        return {"Authorization": f"Bearer {connection.api_key}"} if connection.api_key else {}

    @classmethod
    def body(cls, connection: LlmConnection, model: str, system: str, user: str, max_tokens: int) -> dict[str, Any]:
        """Temperature 0: the same documents give the same answer, which the
        evaluation needs to compare one change with the next."""
        result: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if connection.disable_thinking:
            result["chat_template_kwargs"] = {"enable_thinking": False}
        return result
