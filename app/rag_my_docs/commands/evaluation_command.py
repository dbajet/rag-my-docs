from __future__ import annotations

from rag_my_docs.commands.answer_command import AnswerCommand
from rag_my_docs.constants.evaluation_constants import EvaluationConstants
from rag_my_docs.structures.answer_result import AnswerResult
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.evaluation_case import EvaluationCase
from rag_my_docs.structures.evaluation_outcome import EvaluationOutcome
from rag_my_docs.structures.user_account import UserAccount


class EvaluationCommand:
    """The test set: fixed questions with what a right answer needs, run one
    at a time through the same flow as any question, so a change to
    retrieval, the prompt, the model or the number of passages is measured
    instead of guessed. One question per request: each takes as long as an
    answer."""

    def __init__(self, answers: AnswerCommand, cases: list[EvaluationCase]) -> None:
        self._answers = answers
        self._cases = cases

    def cases(self) -> list[EvaluationCase]:
        return self._cases

    def evaluated(self, user: UserAccount, index: int, top_k: int) -> EvaluationOutcome:
        if not 0 <= index < len(self._cases):
            raise AppException(404, EvaluationConstants.run_out_of_range)
        case = self._cases[index]
        return self.scored(case, self._answers.answered(user, case.question, top_k))

    @classmethod
    def scored(cls, case: EvaluationCase, result: AnswerResult) -> EvaluationOutcome:
        """Retrieval is right when a passage comes from one of the expected
        documents (or none is expected); the answer when it holds every fact,
        or, for a question the publications cannot answer, when it says the
        answer was not found."""
        retrieved = {passage.document_name for passage in result.passages}
        is_retrieval_right = not case.documents or not retrieved.isdisjoint(case.documents)
        text = result.answer.text.casefold()
        missing_facts = tuple(
            fact for fact in case.facts if not any(part.casefold() in text for part in fact.split("|"))
        )
        is_answer_right = (
            result.answer.is_found and not missing_facts if case.expects_answer else not result.answer.is_found
        )
        return EvaluationOutcome(
            question=case.question,
            missing_documents=() if is_retrieval_right else case.documents,
            missing_facts=missing_facts,
            is_retrieval_right=is_retrieval_right,
            is_answer_right=is_answer_right,
            result=result,
        )
