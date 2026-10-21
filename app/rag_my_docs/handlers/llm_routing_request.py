from __future__ import annotations

from pydantic import BaseModel


class LlmRoutingRequest(BaseModel):
    # The connection that answers; null leaves answers without one.
    answers: int | None = None
