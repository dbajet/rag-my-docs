from __future__ import annotations

import re

from rag_my_docs.structures.text_range import TextRange


class SentenceSplitter:
    """Finds sentence boundaries with a few explicit rules, deterministically.

    A boundary is terminal punctuation (. ! ?, with any closing quotes or
    brackets) followed by whitespace and then something that is not a
    lowercase letter. A period is not a boundary after a known abbreviation
    ("Dr.", "e.g.", "Fig."), after a single letter (an initial: "J. Smith"),
    or inside a dotted abbreviation ("U.S."). Decimal numbers ("3.5") never
    split: there is no whitespace after their period. A few abbreviations
    that often end a sentence ("etc.", "a.m.", "Inc.") do end one when what
    follows starts with a capital. A line break inside a paragraph is a list
    item's start, and always a boundary.

    Sentences are returned as ranges of the original text, so nothing is
    rewritten: the whitespace between two ranges is all that is left out."""

    _ABBREVIATIONS = frozenset(
        {
            "mr", "mrs", "ms", "mx", "dr", "prof", "sr", "jr", "st", "mt", "rev", "gen", "col", "capt", "lt",
            "sgt", "gov", "sen", "rep", "hon", "vs", "v", "etc", "e.g", "i.e", "cf", "al", "approx", "ca",
            "fig", "figs", "eq", "eqs", "no", "nos", "vol", "vols", "pp", "p", "ch", "chap", "sec", "art",
            "para", "ed", "eds", "est", "ref", "refs", "dept", "inc", "ltd", "co", "corp", "llc", "jan",
            "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mon", "tue",
            "wed", "thu", "fri", "sat", "sun", "a.m", "p.m", "u.s", "u.k", "e.u", "ph.d", "min", "max",
            "nr", "tel", "ibid", "op", "viz", "resp", "incl", "excl", "dist", "ave", "blvd", "rd",
        }
    )  # fmt: skip
    _MAY_END = frozenset({"etc", "a.m", "p.m", "inc", "ltd", "co", "corp", "llc", "jr", "sr"})
    _END = re.compile(r"[.!?]+[\"'”’)\]]*")
    _WORD_BEFORE = re.compile(r"[\"'(\[“‘]*([^\s\"'(\[“‘]+)$")
    _DOTTED = re.compile(r"(?:[a-z]\.)+[a-z]")

    def sentences(self, text: str) -> list[TextRange]:
        result: list[TextRange] = []
        for line in self.lines(text):
            result.extend(self.in_line(text, line))
        return result

    @classmethod
    def lines(cls, text: str) -> list[TextRange]:
        """The non-blank lines of `text`, trimmed."""
        result: list[TextRange] = []
        start = 0
        for part in text.split("\n"):
            if trimmed := cls.trimmed(text, TextRange(start=start, end=start + len(part))):
                result.append(trimmed)
            start += len(part) + 1
        return result

    def in_line(self, text: str, line: TextRange) -> list[TextRange]:
        result: list[TextRange] = []
        start = line.start
        for match in self._END.finditer(text, line.start, line.end):
            following = self.next_visible(text, match.end(), line.end)
            if following == line.end or following == match.end():
                # The line's own end, or punctuation inside a token ("3.5", "a.b").
                continue
            if self.is_boundary(text, start, match.start(), match.group(0), text[following]):
                result.append(TextRange(start=start, end=match.end()))
                start = following
        if trimmed := self.trimmed(text, TextRange(start=start, end=line.end)):
            result.append(trimmed)
        return result

    def is_boundary(self, text: str, start: int, mark: int, punctuation: str, following: str) -> bool:
        if following.islower():
            return False
        if "!" in punctuation or "?" in punctuation:
            return True
        found = self._WORD_BEFORE.search(text, start, mark)
        word = "" if found is None else found.group(1).lower()
        if word in self._MAY_END:
            return True
        is_initial = len(word) == 1 and word.isalpha()
        return not (word in self._ABBREVIATIONS or is_initial or self._DOTTED.fullmatch(word))

    @classmethod
    def next_visible(cls, text: str, start: int, end: int) -> int:
        """The first index in [start, end) that is not whitespace, or `end`."""
        for index in range(start, end):
            if not text[index].isspace():
                return index
        return end

    @classmethod
    def trimmed(cls, text: str, span: TextRange) -> TextRange | None:
        """The range without its leading and trailing whitespace; None when nothing is left."""
        start = cls.next_visible(text, span.start, span.end)
        end = span.end
        while end > start and text[end - 1].isspace():
            end -= 1
        return TextRange(start=start, end=end) if end > start else None
