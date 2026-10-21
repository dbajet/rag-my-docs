from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _LlmConstants:
    # --- OpenAI-compatible endpoints (vLLM, hosted providers) --------------------------
    chat_path: str = "/chat/completions"
    models_path: str = "/models"

    # --- calls ----------------------------------------------------------------------
    # One call per question, inside the proxy's 150 s for answers
    # (deploy/nginx-rag_my_docs.conf).
    answer_timeout_seconds: float = 120.0
    # A connection's light: an answer within this, or red.
    health_timeout_seconds: float = 8.0
    answer_max_tokens: int = 1500
    # Claude thinks before answering, and its thinking counts against the limit.
    claude_answer_max_tokens: int = 16000
    claude_answer_effort: str = "medium"
    # A declined request is re-run on a model that accepts it, inside the same
    # call, for the models that offer it; others simply report the refusal.
    claude_fallback_beta: str = "server-side-fallback-2026-07-01"
    claude_fallback_models: tuple[str, ...] = (
        "claude-fable-5-1",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
    )
    # The instructions sent with every question; the passages and the
    # question follow in the user message (AnswerCommand.prompt).
    answer_prompt: tuple[str, ...] = (
        "You answer questions about U.S. federal income tax for tax year 2025.",
        "Use only the numbered passages below, taken from IRS publications. Use nothing you know from elsewhere.",
        "If the passages do not contain the answer, say so plainly and set found to false.",
        "Answer briefly, in the language of the question, with the exact figures and words of the passages.",
        'Reply with JSON only: {"found": true or false, "answer": "...", "passages": [the numbers of the passages',
        "the answer comes from]}.",
    )

    # --- connections ---------------------------------------------------------------------
    max_name_chars: int = 80
    # Anthropic's largest page: every model in one request.
    claude_model_page: int = 1000

    # --- messages ---------------------------------------------------------------------
    not_configured: str = "Answers need a language model: choose one in Settings › Language models."
    unavailable: str = "The language model did not answer. Try again in a moment."
    refused: str = "The language model declined to answer this question."
    connection_not_found: str = "This connection does not exist."
    connection_name: str = "Give the connection a name, of at most 80 characters."
    connection_url: str = "An OpenAI-compatible connection needs an http:// or https:// address with a host."
    connection_model: str = "Name the model that answers."
    connection_key: str = "A Claude connection needs an API key."
    health_ok: str = "Connected: the key is accepted and the model is there."
    health_key_refused: str = "The API key was refused."
    health_unreachable: str = "Unreachable: no answer from the address."
    not_a_model_list: str = "The endpoint's answer is not an OpenAI-compatible model list."


LlmConstants = _LlmConstants()
