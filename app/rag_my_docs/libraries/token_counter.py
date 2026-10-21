from __future__ import annotations

import re

from rag_my_docs.structures.text_range import TextRange


class TokenCounter:
    """Counts tokens the simplest explainable way: each run of letters and
    digits (`\\w+`) is one token, and so is each other visible character.
    "Dr. Smith paid $3.50." is 9 tokens: Dr . Smith paid $ 3 . 50 .

    Embedding models cut text into subword tokens instead, about 1.2 to 1.4
    of theirs per one of these in English prose; a 350-token chunk here stays
    well under the 512 tokens most of them accept. When an embedding model is
    chosen, its own tokenizer can replace this class without touching the
    chunker: the chunker only asks for counts and cut points."""

    _TOKEN = re.compile(r"\w+|[^\w\s]")

    @classmethod
    def count(cls, text: str) -> int:
        return sum(1 for _ in cls._TOKEN.finditer(text))

    @classmethod
    def windows(cls, text: str, limit: int) -> list[TextRange]:
        """Consecutive ranges of at most `limit` tokens covering every token of
        `text`, for a sentence too long for one chunk. Each cut falls on
        whitespace when the window has any, so "3.50" or "e-mail" is never cut
        in two; text without a space in `limit` tokens is cut between tokens."""
        tokens = [TextRange(start=match.start(), end=match.end()) for match in cls._TOKEN.finditer(text)]
        result: list[TextRange] = []
        first = 0
        while first < len(tokens):
            last = min(first + limit, len(tokens))
            if last < len(tokens):
                last = cls._cut(tokens, first, last)
            result.append(TextRange(start=tokens[first].start, end=tokens[last - 1].end))
            first = last
        return result

    @classmethod
    def _cut(cls, tokens: list[TextRange], first: int, last: int) -> int:
        """The latest index in (first, last] that starts after whitespace, or `last`."""
        for index in range(last, first, -1):
            if tokens[index].start > tokens[index - 1].end:
                return index
        return last
