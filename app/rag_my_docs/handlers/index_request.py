from __future__ import annotations

from pydantic import BaseModel


class IndexRequest(BaseModel):
    # True embeds every chunk again; False only the chunks without an embedding.
    rebuild: bool = False
