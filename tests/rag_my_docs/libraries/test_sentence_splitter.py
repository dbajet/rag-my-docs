from __future__ import annotations

import random

import pysbd
import pytest

from rag_my_docs.libraries.sentence_splitter import SentenceSplitter
from rag_my_docs.structures.text_range import TextRange


def helper_texts(text: str, ranges: list[TextRange]) -> list[str]:
    return [text[span.start : span.end] for span in ranges]


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("Hello World. My name is Jonas.", ["Hello World.", "My name is Jonas."], id="simple"),
        pytest.param("Dr. Smith paid $3.50. He left.", ["Dr. Smith paid $3.50.", "He left."], id="title-and-decimal"),
        pytest.param("Prices rose 2.5% in May. Sales fell.", ["Prices rose 2.5% in May.", "Sales fell."], id="percent"),
        pytest.param("See Fig. 3 for details. It is clear.", ["See Fig. 3 for details.", "It is clear."], id="fig"),
        pytest.param("He left, e.g. early. Then he came.", ["He left, e.g. early.", "Then he came."], id="e.g."),
        pytest.param("The U.S. Government met. It agreed.", ["The U.S. Government met.", "It agreed."], id="dotted"),
        pytest.param("J. R. R. Tolkien wrote it. Yes.", ["J. R. R. Tolkien wrote it.", "Yes."], id="initials"),
        pytest.param("Is it 3.14? Yes it is! Fine.", ["Is it 3.14?", "Yes it is!", "Fine."], id="question-exclamation"),
        pytest.param(
            '"Really?" she asked. "Yes." He nodded.', ['"Really?" she asked.', '"Yes."', "He nodded."], id="quotes"
        ),
        pytest.param("He said wait. and stopped.", ["He said wait. and stopped."], id="lowercase-continues"),
        pytest.param("We met at 10 a.m. Then we left.", ["We met at 10 a.m.", "Then we left."], id="may-end"),
        pytest.param("We met at 10 a.m. sharp.", ["We met at 10 a.m. sharp."], id="may-end-but-continues"),
        pytest.param("It ends here... Next one.", ["It ends here...", "Next one."], id="ellipsis"),
        pytest.param(
            "Items:\n• First item. Still first\n• Second item",
            ["Items:", "• First item.", "Still first", "• Second item"],
            id="list-lines",
        ),
        pytest.param("  Padded.   Text.  ", ["Padded.", "Text."], id="whitespace-trimmed"),
        pytest.param("", [], id="empty"),
        pytest.param("No end punctuation", ["No end punctuation"], id="no-terminal"),
        pytest.param("(It works.) Then more.", ["(It works.)", "Then more."], id="closing-bracket"),
    ],
)
def test_sentences(text: str, expected: list[str]) -> None:
    tested = SentenceSplitter()
    result = helper_texts(text, tested.sentences(text))
    assert result == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param(
            "a\n  b  \n\n c",
            [TextRange(start=0, end=1), TextRange(start=4, end=5), TextRange(start=10, end=11)],
            id="lines",
        ),
        pytest.param("   ", [], id="blank"),
        pytest.param("one", [TextRange(start=0, end=3)], id="single"),
    ],
)
def test_lines(text: str, expected: list[TextRange]) -> None:
    tested = SentenceSplitter
    result = tested.lines(text)
    assert result == expected


def test_in_line() -> None:
    text = "Skip. Ab cd. Ef gh. Inner 3.5 xy. Skip."
    tested = SentenceSplitter()
    result = helper_texts(text, tested.in_line(text, TextRange(start=6, end=34)))
    expected = ["Ab cd.", "Ef gh.", "Inner 3.5 xy."]
    assert result == expected


@pytest.mark.parametrize(
    "text, start, mark, punctuation, following, expected",
    [
        pytest.param("He left. Then", 0, 7, ".", "T", True, id="period-capital"),
        pytest.param("He left. then", 0, 7, ".", "t", False, id="lowercase-never"),
        pytest.param("Why? Then", 0, 3, "?", "T", True, id="question"),
        pytest.param("Wow!? Then", 0, 3, "!?", "T", True, id="mixed"),
        pytest.param("See Dr. Who", 0, 6, ".", "W", False, id="abbreviation"),
        pytest.param("At 9 a.m. Then", 0, 8, ".", "T", True, id="may-end"),
        pytest.param("By J. Smith", 0, 4, ".", "S", False, id="initial"),
        pytest.param("The U.S.A. Team", 0, 9, ".", "T", False, id="dotted-unknown"),
        pytest.param('("Mr. Smith', 0, 4, ".", "S", False, id="quoted-abbreviation"),
        pytest.param(". Then", 0, 0, ".", "T", True, id="no-word-before"),
        pytest.param("Number 42. Then", 0, 9, ".", "T", True, id="digits-end"),
    ],
)
def test_is_boundary(text: str, start: int, mark: int, punctuation: str, following: str, expected: bool) -> None:
    tested = SentenceSplitter()
    result = tested.is_boundary(text, start, mark, punctuation, following)
    assert result is expected


@pytest.mark.parametrize(
    "text, start, end, expected",
    [
        pytest.param("ab  cd", 2, 6, 4, id="skips-spaces"),
        pytest.param("ab   ", 2, 5, 5, id="none-returns-end"),
        pytest.param("abc", 1, 3, 1, id="already-visible"),
    ],
)
def test_next_visible(text: str, start: int, end: int, expected: int) -> None:
    tested = SentenceSplitter
    result = tested.next_visible(text, start, end)
    assert result == expected


@pytest.mark.parametrize(
    "text, span, expected",
    [
        pytest.param("  ab  ", TextRange(start=0, end=6), TextRange(start=2, end=4), id="both-sides"),
        pytest.param("ab", TextRange(start=0, end=2), TextRange(start=0, end=2), id="unchanged"),
        pytest.param("   ", TextRange(start=0, end=3), None, id="blank"),
        pytest.param("ab", TextRange(start=1, end=1), None, id="empty-range"),
    ],
)
def test_trimmed(text: str, span: TextRange, expected: TextRange | None) -> None:
    tested = SentenceSplitter
    result = tested.trimmed(text, span)
    assert result == expected


# --- comparison with pysbd -------------------------------------------------------------


def helper_pysbd(text: str) -> list[TextRange]:
    """pysbd's sentences as ranges without their trailing whitespace."""
    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    return [TextRange(start=span.start, end=span.start + len(span.sent.rstrip())) for span in segmenter.segment(text)]


AGREED_EXAMPLES = [
    "Hello World. My name is Jonas.",
    "What is your name? My name is Jonas.",
    "There it is! I found it.",
    "My name is Jonas E. Smith.",
    "Please turn to p. 55.",
    "Were Jane and co. at the party?",
    "They closed the deal with Pitt, Briggs & Co. at noon.",
    "Let's ask Jane and co. They should know.",
    "They closed the deal with Pitt, Briggs & Co. It closed yesterday.",
    "I can see Mt. Fuji from here.",
    "St. Michael's Church is on 5th st. near the light.",
    "That is JFK Jr.'s book.",
    "I visited the U.S.A. last year.",
    "I work for the U.S. Government in Virginia.",
    "I have lived in the U.S. for 20 years.",
    "She has $100.00 in her bag.",
    "She has $100.00. It is in her bag.",
    "He teaches science (He previously worked for 5 years as an engineer.) at the local University.",
    "Dr. Smith paid 3.50 dollars. He left.",
    "The price rose 2.5% in May. Sales fell.",
    "See Fig. 3 for details. The chart is clear.",
    "Mr. Brown met Mrs. Green. They talked.",
    "This is e.g. an example. Another one follows.",
    "It costs $3.50. Pay now!",
    "Is it 3.14? Yes it is.",
]

# Intended differences, each with its reason: after a dotted abbreviation and
# before a capital, pysbd guesses a sentence end; we keep the sentence going,
# because "the U.S. Government" is far more common in documents than a
# sentence ending on "U.S." — and a missed boundary only makes a chunk a
# sentence longer, while a wrong one cuts a sentence in half.
DIFFERENT_EXAMPLES = [
    ("I live in the E.U. How about you?", ["I live in the E.U. How about you?"]),
    ("I live in the U.S. How about you?", ["I live in the U.S. How about you?"]),
]


def helper_generated() -> list[str]:
    """Seeded sentences of capitalised subjects, abbreviations and decimals."""
    generator = random.Random(20261020)
    subjects = ["The committee", "Our team", "The report", "Dr. Smith", "Mr. Brown", "The board", "Mrs. Green"]
    verbs = ["reviewed", "approved", "measured", "described", "noted", "rejected", "counted"]
    objects = ["the budget", "the results", "the plan", "the survey", "the figures", "the new site"]
    tails = [
        "",
        " by 3.5 percent",
        " at 12.75 dollars",
        " with care, e.g. twice",
        " in 2.5 hours",
        " for the U.S. office",
    ]
    ends = [".", ".", ".", "?", "!"]
    result: list[str] = []
    for _ in range(200):
        count = generator.randint(1, 4)
        sentences = [
            f"{generator.choice(subjects)} {generator.choice(verbs)} {generator.choice(objects)}"
            f"{generator.choice(tails)}{generator.choice(ends)}"
            for _ in range(count)
        ]
        result.append(" ".join(sentences))
    return result


def test_sentences__agrees_with_pysbd() -> None:
    """A hand-written splitter is checked against an established one over
    published-style examples and generated text; the cases where the two
    disagree on purpose are listed above, with what this splitter answers."""
    tested = SentenceSplitter()
    texts = AGREED_EXAMPLES + helper_generated()
    result = [(text, tested.sentences(text)) for text in texts]
    expected = [(text, helper_pysbd(text)) for text in texts]
    assert result == expected
    differences = [helper_texts(text, tested.sentences(text)) for text, _ in DIFFERENT_EXAMPLES]
    exp_differences = [answer for _, answer in DIFFERENT_EXAMPLES]
    assert differences == exp_differences
    pysbd_differs = [helper_texts(text, helper_pysbd(text)) != answer for text, answer in DIFFERENT_EXAMPLES]
    exp_pysbd_differs = [True, True]
    assert pysbd_differs == exp_pysbd_differs
