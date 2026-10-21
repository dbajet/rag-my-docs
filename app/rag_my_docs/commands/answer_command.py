from __future__ import annotations

import json
import time
from typing import Any

from rag_my_docs.constants.answer_constants import AnswerConstants
from rag_my_docs.constants.embedding_constants import EmbeddingConstants
from rag_my_docs.constants.llm_constants import LlmConstants
from rag_my_docs.libraries.embedding_box import EmbeddingBox
from rag_my_docs.libraries.llm_gateway import LlmGateway
from rag_my_docs.libraries.search_store import SearchStore
from rag_my_docs.structures.answer_citation import AnswerCitation
from rag_my_docs.structures.answer_result import AnswerResult
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.model_answer import ModelAnswer
from rag_my_docs.structures.search_hit import SearchHit
from rag_my_docs.structures.user_account import UserAccount


class AnswerCommand:
    """Retrieval-augmented generation, in its classic form:

    1. the question is embedded by the same local model as the passages
       (vectors from two models are not comparable);
    2. the `top_k` passages closest to it are retrieved, by cosine similarity;
    3. when even the closest is far from the question, the publications do
       not cover it, and no model is called;
    4. otherwise the passages, numbered, and the question are sent to the
       language model, which answers from them alone and names the passages
       it used; their publications and pages become the citations.

    Every step is returned — the passages with their scores, the exact
    prompt, the answer — so the page can show what retrieval did, and what
    the model was given."""

    def __init__(self, llm: LlmGateway, store: SearchStore, embeddings: EmbeddingBox) -> None:
        self._llm = llm
        self._store = store
        self._embeddings = embeddings

    def answered(self, user: UserAccount, question: str, top_k: int) -> AnswerResult:
        asked = self.checked(question, top_k)
        if not self._llm.is_configured():
            raise AppException(503, LlmConstants.not_configured)
        space = self._embeddings.space()
        if self._store.indexed_count(user.user_id, space) == 0:
            raise AppException(409, EmbeddingConstants.nothing_indexed)
        started = time.perf_counter()
        vector = self._embeddings.query(asked)
        embedded = time.perf_counter()
        passages = self._store.nearest(user.user_id, vector, space, top_k)
        searched = time.perf_counter()
        embedding_ms = round((embedded - started) * 1000, 1)
        search_ms = round((searched - embedded) * 1000, 1)
        if not passages or passages[0].similarity < AnswerConstants.min_similarity:
            answer = ModelAnswer(is_found=False, text=AnswerConstants.not_covered, citations=())
            return AnswerResult(asked, tuple(passages), "", "", answer, "", embedding_ms, search_ms, 0.0)
        system = "\n".join(LlmConstants.answer_prompt)
        prompt = self.prompt(passages, asked)
        reply, model = self._llm.answer_reply(system, prompt)
        return AnswerResult(
            question=asked,
            passages=tuple(passages),
            system=system,
            prompt=prompt,
            answer=self.parsed_answer(reply, passages),
            model=model,
            embedding_ms=embedding_ms,
            search_ms=search_ms,
            answer_ms=round((time.perf_counter() - searched) * 1000, 1),
        )

    @classmethod
    def checked(cls, question: str, top_k: int) -> str:
        """The question, trimmed; refused when empty, too long, or with an impossible number of passages."""
        result = question.strip()
        if not result:
            raise AppException(400, AnswerConstants.empty_question)
        if len(result) > EmbeddingConstants.max_query_chars:
            raise AppException(400, f"A question has at most {EmbeddingConstants.max_query_chars} characters.")
        if not 1 <= top_k <= EmbeddingConstants.max_top_k:
            raise AppException(400, f"The number of passages is between 1 and {EmbeddingConstants.max_top_k}.")
        return result

    @classmethod
    def prompt(cls, passages: list[SearchHit], question: str) -> str:
        """What the model reads after its instructions: each passage under its
        number, its publication, section and pages, then the question."""
        lines: list[str] = []
        for passage in passages:
            lines.append(f"[{passage.rank}] {passage.document_name} › {passage.section_path}, {cls.pages(passage)}")
            lines.append(passage.text)
            lines.append("")
        lines.append(f"Question: {question}")
        return "\n".join(lines)

    @classmethod
    def pages(cls, passage: SearchHit) -> str:
        if passage.start_page == passage.end_page:
            return f"page {passage.start_page}"
        return f"pages {passage.start_page}–{passage.end_page}"

    @classmethod
    def parsed_answer(cls, reply: str, passages: list[SearchHit]) -> ModelAnswer:
        """The model's JSON answer, its passage numbers turned into citations
        (a number no passage has is dropped); its plain text, taken as found
        and uncited, when it answered without the JSON asked for."""
        try:
            data = cls.json_object(reply)
            return ModelAnswer(
                is_found=data.get("found") is not False,
                text=str(data["answer"]).strip(),
                citations=cls.citations(data.get("passages"), passages),
            )
        except (ValueError, KeyError, TypeError):
            return ModelAnswer(is_found=True, text=reply.strip(), citations=())

    @classmethod
    def citations(cls, numbers: Any, passages: list[SearchHit]) -> tuple[AnswerCitation, ...]:
        """The passages named, once each, in the order named; nothing when
        the model gave no list."""
        if not isinstance(numbers, list):
            return ()
        by_rank = {passage.rank: passage for passage in passages}
        named = [number for number in numbers if isinstance(number, int) and not isinstance(number, bool)]
        return tuple(
            AnswerCitation(number, by_rank[number].document_name, by_rank[number].start_page)
            for number in dict.fromkeys(named)
            if number in by_rank
        )

    @classmethod
    def json_object(cls, reply: str) -> dict[str, Any]:
        """The JSON object in a reply, fenced or wrapped in prose as models do."""
        start = reply.find("{")
        end = reply.rfind("}")
        if start < 0 or end < start:
            raise ValueError("no JSON object")
        # From "{" to "}": valid JSON there can only be an object.
        result: dict[str, Any] = json.loads(reply[start : end + 1])
        return result
