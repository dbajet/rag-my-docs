from __future__ import annotations

from typing import Any, NamedTuple


class Publication(NamedTuple):
    """One document of the corpus: the name it is shown and cited under,
    where it is downloaded from, and the SHA-256 the download must have."""

    name: str
    url: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "url": self.url, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Publication:
        return cls(name=str(data["name"]), url=str(data["url"]), sha256=str(data["sha256"]))
