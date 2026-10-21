from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from rag_my_docs.commands.answer_command import AnswerCommand
from rag_my_docs.commands.corpus_command import CorpusCommand
from rag_my_docs.commands.document_command import DocumentCommand
from rag_my_docs.commands.evaluation_command import EvaluationCommand
from rag_my_docs.commands.health_command import HealthCommand
from rag_my_docs.commands.index_command import IndexCommand
from rag_my_docs.commands.llm_connection_command import LlmConnectionCommand
from rag_my_docs.commands.owner_command import OwnerCommand
from rag_my_docs.commands.processing_command import ProcessingCommand
from rag_my_docs.commands.session_command import SessionCommand
from rag_my_docs.commands.sign_in_command import SignInCommand
from rag_my_docs.constants.constants import Constants
from rag_my_docs.constants.corpus_constants import CorpusConstants
from rag_my_docs.constants.evaluation_constants import EvaluationConstants
from rag_my_docs.handlers.account_router import AccountRouter
from rag_my_docs.handlers.answer_router import AnswerRouter
from rag_my_docs.handlers.api_router import ApiRouter
from rag_my_docs.handlers.auth_router import AuthRouter
from rag_my_docs.handlers.document_router import DocumentRouter
from rag_my_docs.handlers.evaluation_router import EvaluationRouter
from rag_my_docs.handlers.index_router import IndexRouter
from rag_my_docs.handlers.llm_connection_router import LlmConnectionRouter
from rag_my_docs.handlers.processing_router import ProcessingRouter
from rag_my_docs.handlers.session_guard import SessionGuard
from rag_my_docs.libraries.alerter import Alerter
from rag_my_docs.libraries.blind_index import BlindIndex
from rag_my_docs.libraries.chunker import Chunker
from rag_my_docs.libraries.claude_box import ClaudeBox
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.document_store import DocumentStore
from rag_my_docs.libraries.email_box import EmailBox
from rag_my_docs.libraries.email_composer import EmailComposer
from rag_my_docs.libraries.embedding_box import EmbeddingBox
from rag_my_docs.libraries.index_store import IndexStore
from rag_my_docs.libraries.job_queue import JobQueue
from rag_my_docs.libraries.llm_connection_store import LlmConnectionStore
from rag_my_docs.libraries.llm_gateway import LlmGateway
from rag_my_docs.libraries.magic_link_store import MagicLinkStore
from rag_my_docs.libraries.openai_chat_box import OpenAiChatBox
from rag_my_docs.libraries.page_furniture import PageFurniture
from rag_my_docs.libraries.page_layout import PageLayout
from rag_my_docs.libraries.page_renderer import PageRenderer
from rag_my_docs.libraries.pdf_box import PdfBox
from rag_my_docs.libraries.processing_store import ProcessingStore
from rag_my_docs.libraries.publication_box import PublicationBox
from rag_my_docs.libraries.search_store import SearchStore
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.section_detector import SectionDetector
from rag_my_docs.libraries.section_paths import SectionPaths
from rag_my_docs.libraries.security_headers import SecurityHeaders
from rag_my_docs.libraries.sentence_splitter import SentenceSplitter
from rag_my_docs.libraries.session_store import SessionStore
from rag_my_docs.libraries.settings_loader import SettingsLoader
from rag_my_docs.libraries.text_digest import TextDigest
from rag_my_docs.libraries.token_counter import TokenCounter
from rag_my_docs.libraries.user_store import UserStore
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.evaluation_case import EvaluationCase
from rag_my_docs.structures.publication import Publication
from rag_my_docs.structures.settings import Settings


class AppFactory:
    """The composition root: the one place that builds the boxes and
    the commands and hands them to each other."""

    def create(self) -> FastAPI:
        return self.created(SettingsLoader.of_process().loaded())

    def created(self, settings: Settings) -> FastAPI:
        return self.built(settings, [Publication(*publication) for publication in CorpusConstants.publications])

    def built(self, settings: Settings, publications: list[Publication]) -> FastAPI:
        database = Database(settings.database_url)
        email_box = EmailBox(settings)
        alerter = Alerter(email_box, settings.bootstrap_admin_email)
        jobs = JobQueue("mail", alerter)
        # Its own thread: a long PDF must not hold back a sign-in email.
        processing_jobs = JobQueue("processing", alerter)
        # And one for embeddings: a document being embedded does not hold back the next extraction.
        index_jobs = JobQueue("indexing", alerter)
        # And one for downloads: the first publication is processed while the next downloads.
        corpus_jobs = JobQueue("corpus", alerter)
        sealer = SecretSealer(settings.data_key)
        users = UserStore(database, sealer, BlindIndex(settings.data_key))
        session_store = SessionStore(database)
        document_store = DocumentStore(database, sealer)
        sessions = SessionCommand(users, session_store)
        sign_in = SignInCommand(settings, users, MagicLinkStore(database), sessions, EmailComposer(), email_box, jobs)
        embeddings = EmbeddingBox(settings.embedding)
        paths = SectionPaths(database, sealer)
        indexing = IndexCommand(IndexStore(database, sealer, paths), embeddings, index_jobs)
        connection_store = LlmConnectionStore(database, sealer)
        gateway = LlmGateway(connection_store, OpenAiChatBox(), ClaudeBox())
        answers = AnswerCommand(gateway, SearchStore(database, sealer, paths), embeddings)
        processing = ProcessingCommand(
            ProcessingStore(database, sealer, TextDigest(settings.data_key)),
            PdfBox(),
            PageLayout(),
            PageFurniture(),
            SectionDetector(),
            Chunker(settings.chunking, SentenceSplitter(), TokenCounter()),
            settings.chunking.strategy(),
            processing_jobs,
            indexing,
        )
        corpus = CorpusCommand(publications, PublicationBox(), document_store, processing, corpus_jobs)
        evaluations = EvaluationCommand(answers, [EvaluationCase(*case) for case in EvaluationConstants.cases])
        guard = SessionGuard(settings, sessions)
        static_directory = Path(__file__).parent / Constants.static_directory
        routers = [
            AuthRouter(guard, sign_in, sessions).router(),
            AccountRouter(guard).router(),
            DocumentRouter(guard, DocumentCommand(document_store)).router(),
            ProcessingRouter(guard, processing).router(),
            IndexRouter(guard, indexing).router(),
            AnswerRouter(guard, answers).router(),
            LlmConnectionRouter(guard, LlmConnectionCommand(connection_store, gateway)).router(),
            EvaluationRouter(guard, evaluations).router(),
            ApiRouter(HealthCommand(settings, database), PageRenderer(static_directory, settings)).router(),
        ]
        result = FastAPI(
            lifespan=self.lifespan(
                settings,
                database,
                OwnerCommand(users),
                corpus,
                processing,
                indexing,
                [jobs, processing_jobs, index_jobs, corpus_jobs],
            ),
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )
        result.add_middleware(SecurityHeaders)
        result.add_exception_handler(AppException, self.app_exception_handler)
        for router in routers:
            result.include_router(router)
        result.mount(Constants.static_mount, StaticFiles(directory=static_directory), name="static")
        return result

    @classmethod
    def lifespan(
        cls,
        settings: Settings,
        database: Database,
        owners: OwnerCommand,
        corpus: CorpusCommand,
        processing: ProcessingCommand,
        indexing: IndexCommand,
        job_queues: list[JobQueue],
    ) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
        """What a color does when it starts and when it stops."""

        @asynccontextmanager
        async def result(_: FastAPI) -> AsyncIterator[None]:
            # The schema is brought up to date before the first request.
            applied = database.migrated()
            if applied:
                logging.getLogger(Constants.logger_name).info("[DATABASE] migrations applied: %s", applied)
            owner = owners.owner(settings.bootstrap_admin_email)
            for queue in job_queues:
                queue.started()
            # Publications not stored yet are downloaded, then processed and embedded.
            if owner is not None:
                corpus.queued(owner)
            # Documents left waiting by the previous process are queued again.
            processing.resumed()
            # The model loads on the indexing thread, then waiting documents are embedded.
            indexing.resumed()
            yield
            # Queued emails leave, and the running document finishes, before the color stops.
            for queue in job_queues:
                queue.stopped()
            database.closed()

        return result

    @classmethod
    async def app_exception_handler(cls, _: Request, exception: Exception) -> JSONResponse:
        # Starlette's handler signature takes any Exception; only AppException is registered here.
        if not isinstance(exception, AppException):
            raise exception
        return JSONResponse({"detail": exception.message}, status_code=exception.status_code)


# FastAPI needs a module-level ASGI callable for uvicorn discovery.
app = AppFactory().create()
