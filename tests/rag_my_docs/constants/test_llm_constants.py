from __future__ import annotations

import dataclasses

from rag_my_docs.constants.llm_constants import LlmConstants
from tests.conftest import is_dataclass


def test_class() -> None:
    tested = LlmConstants
    fields = {
        "chat_path": "str",
        "models_path": "str",
        "answer_timeout_seconds": "float",
        "health_timeout_seconds": "float",
        "answer_max_tokens": "int",
        "claude_answer_max_tokens": "int",
        "claude_answer_effort": "str",
        "claude_fallback_beta": "str",
        "claude_fallback_models": "tuple[str, ...]",
        "answer_prompt": "tuple[str, ...]",
        "max_name_chars": "int",
        "claude_model_page": "int",
        "not_configured": "str",
        "unavailable": "str",
        "refused": "str",
        "connection_not_found": "str",
        "connection_name": "str",
        "connection_url": "str",
        "connection_model": "str",
        "connection_key": "str",
        "health_ok": "str",
        "health_key_refused": "str",
        "health_unreachable": "str",
        "not_a_model_list": "str",
    }
    assert is_dataclass(tested, fields)


def test_values() -> None:
    tested = LlmConstants
    result = dataclasses.asdict(tested)
    expected = {
        "chat_path": "/chat/completions",
        "models_path": "/models",
        "answer_timeout_seconds": 120.0,
        "health_timeout_seconds": 8.0,
        "answer_max_tokens": 1500,
        "claude_answer_max_tokens": 16000,
        "claude_answer_effort": "medium",
        "claude_fallback_beta": "server-side-fallback-2026-07-01",
        "claude_fallback_models": ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"),
        "answer_prompt": (
            "You answer questions about U.S. federal income tax for tax year 2025.",
            "Use only the numbered passages below, taken from IRS publications. Use nothing you know from elsewhere.",
            "If the passages do not contain the answer, say so plainly and set found to false.",
            "Answer briefly, in the language of the question, with the exact figures and words of the passages.",
            'Reply with JSON only: {"found": true or false, "answer": "...", "passages": [the numbers of the passages',
            "the answer comes from]}.",
        ),
        "max_name_chars": 80,
        "claude_model_page": 1000,
        "not_configured": "Answers need a language model: choose one in Settings › Language models.",
        "unavailable": "The language model did not answer. Try again in a moment.",
        "refused": "The language model declined to answer this question.",
        "connection_not_found": "This connection does not exist.",
        "connection_name": "Give the connection a name, of at most 80 characters.",
        "connection_url": "An OpenAI-compatible connection needs an http:// or https:// address with a host.",
        "connection_model": "Name the model that answers.",
        "connection_key": "A Claude connection needs an API key.",
        "health_ok": "Connected: the key is accepted and the model is there.",
        "health_key_refused": "The API key was refused.",
        "health_unreachable": "Unreachable: no answer from the address.",
        "not_a_model_list": "The endpoint's answer is not an OpenAI-compatible model list.",
    }
    assert result == expected
