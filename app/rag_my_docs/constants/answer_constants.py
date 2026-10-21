from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _AnswerConstants:
    # A question whose closest passage is less similar than this (cosine) is
    # not about the publications: it is answered "not covered" without
    # calling the model. Measured with bge-small on the corpus: the test
    # set's questions score 0.73 to 0.89, questions on other subjects (a
    # recipe, a car's price, sport) 0.46 to 0.65; a tax question the
    # publications do not answer scores in between and goes to the model,
    # which says so.
    min_similarity: float = 0.66
    empty_question: str = "Type a question to answer."
    not_covered: str = "The publications do not cover this question: no passage is close enough to it."


AnswerConstants = _AnswerConstants()
