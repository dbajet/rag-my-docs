from __future__ import annotations

from typing import Any, NamedTuple

from rag_my_docs.constants.media_type import MediaType


class DocumentFile(NamedTuple):
    """A document opened for download: the name and type it is served under,
    and its content in the clear."""

    name: str
    media_type: MediaType
    content: bytes

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "media_type": self.media_type.value, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentFile:
        return cls(
            name=str(data["name"]),
            media_type=MediaType(data["media_type"]),
            content=bytes(data["content"]),
        )
