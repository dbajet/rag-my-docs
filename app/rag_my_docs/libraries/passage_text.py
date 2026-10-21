from __future__ import annotations

from pathlib import Path

from rag_my_docs.constants.embedding_constants import EmbeddingConstants


class PassageText:
    """What is embedded for a chunk: the document's name and the section path
    above the chunk's own text. A chunk alone often lacks its subject — a row
    of a specifications table never says which laptop it is about — and the
    header puts it back. The text stored and shown is the chunk's own; only its
    vector carries the header. Changing this format changes every vector, so it
    is versioned (`EmbeddingConstants.index_format`)."""

    @classmethod
    def of(cls, document_name: str, section_path: str, text: str) -> str:
        title = Path(document_name).stem
        return f"{title}{EmbeddingConstants.path_separator}{section_path}\n\n{text}"
