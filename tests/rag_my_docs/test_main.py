from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import call, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from rag_my_docs.commands.corpus_command import CorpusCommand
from rag_my_docs.commands.index_command import IndexCommand
from rag_my_docs.commands.owner_command import OwnerCommand
from rag_my_docs.commands.processing_command import ProcessingCommand
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.libraries.blind_index import BlindIndex
from rag_my_docs.libraries.database import Database
from rag_my_docs.libraries.document_store import DocumentStore
from rag_my_docs.libraries.job_queue import JobQueue
from rag_my_docs.libraries.secret_sealer import SecretSealer
from rag_my_docs.libraries.settings_loader import SettingsLoader
from rag_my_docs.libraries.user_store import UserStore
from rag_my_docs.main import AppFactory, app
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.chunking_settings import ChunkingSettings
from rag_my_docs.structures.embedding_settings import EmbeddingSettings
from rag_my_docs.structures.publication import Publication
from rag_my_docs.structures.settings import Settings
from rag_my_docs.structures.user_account import UserAccount
from tests.conftest import helper_database
from tests.support.fake_irs import FakeIrs
from tests.support.fake_openai_chat import FakeOpenAiChat
from tests.support.fake_sentence_transformer import FakeSentenceTransformer
from tests.support.fake_smtp import FakeSMTP
from tests.support.pdf_factory import PdfFactory

INTRO = "This report explains the yearly results. Dr. Smith led the work in 2026. Costs rose by 3.5 percent."
SCOPE = "The scope covers two regions. Each region reports its own sales figures every quarter."
RESULTS = "Sales grew in both regions. The north grew faster than the south. Margins stayed stable."
DEFAULT_STRATEGY = "sentences-v2 target=200 max=350 min=75 overlap=1"
SMALL_STRATEGY = "sentences-v2 target=8 max=12 min=2 overlap=0"


def helper_settings() -> Settings:
    return helper_settings_with(ChunkingSettings(target_tokens=200, max_tokens=350, min_tokens=75, overlap_sentences=1))


def helper_settings_with(chunking: ChunkingSettings) -> Settings:
    return Settings(
        environment="dev",
        database_url=os.environ["TEST_DATABASE_URL"],
        app_version="2026-10-19.1",
        build_time="20261019-120000Z",
        app_commit="5ca1ab1e",
        app_base_url="http://testserver",
        data_key="Zm9yLXRlc3RzLW9ubHktZm9yLXRlc3RzLW9ubHktMzI=",
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="relay-user",
        smtp_password="relay-secret",
        email_from="docs@example.com",
        bootstrap_admin_email="boss@example.com",
        chunking=chunking,
        embedding=EmbeddingSettings(
            model="BAAI/bge-small-en-v1.5",
            query_prefix="Represent this sentence for searching relevant passages: ",
            batch_size=2,
        ),
    )


def helper_report_pdf() -> bytes:
    """Two pages, three numbered headings, one nested: three sections."""
    return (
        PdfFactory()
        .page()
        .paragraph("1. Introduction", 14, True)
        .paragraph(INTRO, 10, False)
        .paragraph("1.1 Scope", 12, True)
        .paragraph(SCOPE, 10, False)
        .page()
        .paragraph("2. Results", 14, True)
        .paragraph(RESULTS, 10, False)
        .built()
    )


def helper_settled(client: TestClient, document_id: int) -> dict[str, Any]:
    """The document's processing once it was downloaded and has left both
    queues — extraction, then indexing — or after ten seconds: the threads
    are real, so the test waits for them."""
    waiter = threading.Event()
    result: dict[str, Any] = {}
    for _ in range(200):
        response = client.get(f"/api/documents/{document_id}/processing")
        result = response.json()
        document = result.get("document", {})
        if (
            response.status_code == 200
            and document["status"] not in ("pending", "processing")
            and document["index_status"] not in ("pending", "indexing")
        ):
            return result
        waiter.wait(0.05)
    return result


def helper_publication(name: str, content: bytes) -> Publication:
    """A publication of the test corpus, served by the fake IRS at a URL of its name."""
    return Publication(name, f"https://www.irs.gov/pub/irs-pdf/{name}", hashlib.sha256(content).hexdigest())


def helper_irs(files: dict[str, bytes]) -> FakeIrs:
    return FakeIrs({f"https://www.irs.gov/pub/irs-pdf/{name}": content for name, content in files.items()})


def helper_without_ids(processing: dict[str, Any]) -> dict[str, Any]:
    """Sections and chunks without their database ids, which a new run renumbers."""
    positions = {section["section_id"]: section["position"] for section in processing["sections"]}
    return {
        "sections": [
            {**section, "section_id": None, "parent_id": positions.get(section["parent_id"])}
            for section in processing["sections"]
        ],
        "chunks": [
            {**chunk, "chunk_id": None, "section_id": positions[chunk["section_id"]]} for chunk in processing["chunks"]
        ],
    }


def helper_row_counts(database: Database) -> dict[str, int]:
    row = database.row(
        """SELECT (SELECT count(*) FROM document_sections) AS sections,
                  (SELECT count(*) FROM document_chunks) AS chunks""",
        (),
    )
    return {"sections": int(row["sections"]), "chunks": int(row["chunks"])} if row is not None else {}


def helper_origin() -> dict[str, str]:
    return {"Origin": "http://testserver"}


def helper_sent(fake: FakeSMTP, count: int) -> list[tuple[Any, ...]]:
    """The messages the job thread handed to the relay, once `count` of them
    are there or after five seconds: the thread is real, so the test waits."""
    waiter = threading.Event()
    for _ in range(500):
        result = [entry for entry in fake.calls if entry[0] == "send_message"]
        if len(result) >= count:
            return result
        waiter.wait(0.01)
    return [entry for entry in fake.calls if entry[0] == "send_message"]


def helper_token(message: tuple[Any, ...]) -> str:
    found = re.search(r"/auth/link\?token=([A-Za-z0-9_-]+)", str(message[4]))
    return "" if found is None else found.group(1)


def helper_masked(message: tuple[Any, ...]) -> tuple[Any, ...]:
    """The message with its random token replaced, so the rest is asserted whole."""
    return (*message[:4], re.sub(r"token=[A-Za-z0-9_-]+", "token=<token>", str(message[4])))


def helper_signed_in(client: TestClient, message: tuple[Any, ...]) -> int:
    client.cookies.clear()
    return client.post("/api/auth/link", json={"token": helper_token(message)}, headers=helper_origin()).status_code


def helper_owner_in(client: TestClient, fake: FakeSMTP, count: int) -> int:
    """The owner signed in through the `count`-th email sent."""
    client.post("/api/auth/link-request", json={"email": "boss@example.com"}, headers=helper_origin())
    return helper_signed_in(client, helper_sent(fake, count)[count - 1])


def helper_unstamped(body: Any) -> Any:
    """The answer with each `YYYY-MM-DDTHH:MM:SSZ` time replaced, so the rest is asserted whole."""
    text = json.dumps(body, ensure_ascii=False)
    return json.loads(re.sub(r'"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"', '"<time>"', text))


def helper_untimed(processing: dict[str, Any]) -> dict[str, Any]:
    """The processing with the indexing run's duration replaced, so the rest is asserted whole."""
    document = processing["document"]
    return processing | {"document": document | {"index_report": document["index_report"] | {"duration_ms": "<ms>"}}}


def helper_unscored(answer: dict[str, Any]) -> dict[str, Any]:
    """The answer with its similarities and durations replaced, so the rest is asserted whole."""
    return answer | {
        "passages": [passage | {"similarity": "<similarity>"} for passage in answer["passages"]],
        "embedding_ms": "<ms>",
        "search_ms": "<ms>",
        "answer_ms": "<ms>",
    }


def helper_routes(built: FastAPI) -> list[tuple[str, tuple[str, ...]]]:
    """Every route the application answers, with the routers it includes
    opened up: FastAPI keeps an included router as one entry of its own."""
    result: list[tuple[str, tuple[str, ...]]] = []
    for entry in built.routes:
        routes = entry.original_router.routes if hasattr(entry, "original_router") else [entry]
        result.extend(
            (route.path, tuple(sorted(getattr(route, "methods", None) or ())))  # type: ignore[attr-defined]
            for route in routes
        )
    return result


def helper_request() -> Request:
    return Request({"type": "http", "method": "GET", "path": "/", "headers": []})


def test_app() -> None:
    result = isinstance(app, FastAPI)
    assert result is True


def test_create() -> None:
    tested = AppFactory()
    settings = helper_settings()
    built = FastAPI()
    with (
        patch.object(SettingsLoader, "loaded", side_effect=[settings]) as mock_loaded,
        patch.object(AppFactory, "created", side_effect=[built]) as mock_created,
    ):
        result = tested.create()
    assert result is built
    exp_calls = [call()]
    assert mock_loaded.mock_calls == exp_calls
    exp_calls = [call(settings)]
    assert mock_created.mock_calls == exp_calls


def test_created() -> None:
    """The corpus is the list in CorpusConstants, as publications."""
    tested = AppFactory()
    settings = helper_settings()
    built = FastAPI()
    corpus = SimpleNamespace(publications=(("Publication 17", "https://www.irs.gov/pub/irs-pdf/p17.pdf", "ab12"),))
    with (
        patch("rag_my_docs.main.CorpusConstants", new=corpus),
        patch.object(AppFactory, "built", side_effect=[built]) as mock_built,
    ):
        result = tested.created(settings)
    assert result is built
    exp_calls = [call(settings, [Publication("Publication 17", "https://www.irs.gov/pub/irs-pdf/p17.pdf", "ab12")])]
    assert mock_built.mock_calls == exp_calls


def test_built() -> None:
    tested = AppFactory()
    built = tested.built(helper_settings(), [])
    result = sorted((path, methods) for path, methods in helper_routes(built))
    expected = [
        ("/", ("GET", "HEAD")),
        ("/api/account", ("GET",)),
        ("/api/answers", ("POST",)),
        ("/api/auth/link", ("POST",)),
        ("/api/auth/link-request", ("POST",)),
        ("/api/auth/sign-out", ("POST",)),
        ("/api/documents", ("GET",)),
        ("/api/documents/{document_id}/content", ("GET",)),
        ("/api/documents/{document_id}/extraction", ("GET",)),
        ("/api/documents/{document_id}/index", ("POST",)),
        ("/api/documents/{document_id}/processing", ("GET",)),
        ("/api/documents/{document_id}/processing", ("POST",)),
        ("/api/evaluation/cases", ("GET",)),
        ("/api/evaluation/run", ("POST",)),
        ("/api/index/all", ("POST",)),
        ("/api/llm/connections", ("GET",)),
        ("/api/llm/connections", ("POST",)),
        ("/api/llm/connections/{connection_id}", ("DELETE",)),
        ("/api/llm/connections/{connection_id}", ("PUT",)),
        ("/api/llm/connections/{connection_id}/health", ("GET",)),
        ("/api/llm/models", ("POST",)),
        ("/api/llm/routing", ("PUT",)),
        ("/auth/link", ("GET", "HEAD")),
        ("/healthz", ("GET", "HEAD")),
        ("/static", ()),
        ("/sw.js", ("GET",)),
    ]
    assert result == expected
    exp_flags = (None, None, None)
    assert (built.docs_url, built.redoc_url, built.openapi_url) == exp_flags


@pytest.mark.database
def test_created__the_shell_and_its_health(caplog: pytest.LogCaptureFixture) -> None:
    """The wiring, unmocked: lifespan on, the schema and the owner's
    account in place before the first request."""
    helper_database()
    tested = AppFactory()
    with (
        caplog.at_level(logging.INFO, logger="rag_my_docs"),
        FakeSentenceTransformer().installed(),
        TestClient(tested.built(helper_settings(), [])) as client,
    ):
        health = client.get("/healthz")
        home = client.get("/")
        link_page = client.get("/auth/link")
        missing = client.get("/no-such-page")
    result = [
        (health.status_code, health.json(), health.headers["x-frame-options"]),
        (home.status_code, home.headers["content-type"], home.headers["cache-control"]),
        (link_page.status_code, link_page.headers["content-type"]),
        (missing.status_code, missing.json()),
    ]
    expected = [
        (200, {"ok": True, "version": "2026-10-19.1", "commit": "5ca1ab1e"}, "DENY"),
        (200, "text/html; charset=utf-8", "no-cache"),
        (200, "text/html; charset=utf-8"),
        (404, {"detail": "Not Found"}),
    ]
    assert result == expected
    exp_messages: list[str] = []
    assert [record.getMessage() for record in caplog.records] == exp_messages


@pytest.mark.database
def test_created__sign_in_flow() -> None:
    """Sign in end to end: the owner asks for a link, the job thread mails
    it through the relay, the link opens a session once."""
    helper_database()
    tested = AppFactory()
    fake = FakeSMTP()
    with (
        fake.installed(),
        FakeSentenceTransformer().installed(),
        TestClient(tested.built(helper_settings(), [])) as client,
    ):
        before = client.get("/api/account")
        requested = client.post("/api/auth/link-request", json={"email": "Boss@Example.com "}, headers=helper_origin())
        message = helper_sent(fake, 1)[0]
        redeemed = client.post("/api/auth/link", json={"token": helper_token(message)}, headers=helper_origin())
        cookie = redeemed.headers["set-cookie"].split("=", 1)[0]
        me = client.get("/api/account")
        reused = client.post("/api/auth/link", json={"token": helper_token(message)}, headers=helper_origin())
    result = [
        (before.status_code, before.json()),
        (requested.status_code, requested.json()),
        (redeemed.status_code, redeemed.json(), cookie),
        (me.status_code, me.json()),
        (reused.status_code, reused.json()),
    ]
    expected = [
        (401, {"detail": "Please sign in."}),
        (200, {"message": "If this address has access, a sign-in link is on its way."}),
        (200, {"ok": True}, "rag_my_docs_session"),
        (
            200,
            {
                "user_id": 1,
                "email": "boss@example.com",
                "display_name": "boss",
                "is_disabled": False,
            },
        ),
        (401, {"detail": "This sign-in link has expired or was already used. Ask for a new one."}),
    ]
    assert result == expected
    exp_calls = [
        ("connect", "smtp.example.com", 587, 15.0),
        ("starttls", "CERT_REQUIRED", True),
        ("login", "relay-user", "relay-secret"),
        helper_masked(message),
        ("quit",),
    ]
    assert [helper_masked(entry) if entry[0] == "send_message" else entry for entry in fake.calls] == exp_calls
    exp_message = (
        "send_message",
        "docs@example.com",
        "boss@example.com",
        "Your RAG My Docs sign-in link",
        "Hello boss,\n"
        "\n"
        "To sign in to RAG My Docs, open this link:\n"
        "http://testserver/auth/link?token=<token>\n"
        "\n"
        "It works once, for 15 minutes.\n"
        "If you did not ask for it, ignore this message.\n",
    )
    assert helper_masked(message) == exp_message


@pytest.mark.database
def test_created__unknown_address_gets_the_same_answer_and_no_email() -> None:
    helper_database()
    tested = AppFactory()
    fake = FakeSMTP()
    with (
        fake.installed(),
        FakeSentenceTransformer().installed(),
        TestClient(tested.built(helper_settings(), [])) as client,
    ):
        unknown = client.post("/api/auth/link-request", json={"email": "nobody@example.com"}, headers=helper_origin())
        known = client.post("/api/auth/link-request", json={"email": "boss@example.com"}, headers=helper_origin())
        # The known address's email marks the queue as drained past the unknown one.
        sent = helper_sent(fake, 1)
    result = [(unknown.status_code, unknown.json()), (known.status_code, known.json()), [entry[2] for entry in sent]]
    expected = [
        (200, {"message": "If this address has access, a sign-in link is on its way."}),
        (200, {"message": "If this address has access, a sign-in link is on its way."}),
        ["boss@example.com"],
    ]
    assert result == expected


@pytest.mark.database
def test_created__a_request_from_another_site_is_refused() -> None:
    helper_database()
    tested = AppFactory()
    with FakeSentenceTransformer().installed(), TestClient(tested.built(helper_settings(), [])) as client:
        response = client.post(
            "/api/auth/link-request",
            json={"email": "boss@example.com"},
            headers={"Origin": "https://evil.example.com"},
        )
    result = (response.status_code, response.json())
    expected = (403, {"detail": "This request did not come from RAG My Docs."})
    assert result == expected


@pytest.mark.database
def test_created__corpus_flow(caplog: pytest.LogCaptureFixture) -> None:
    """At startup the corpus is loaded for the owner: a stored document the
    list does not name is deleted, a publication is downloaded, checked and
    stored sealed, one the server lacks or whose checksum changed is logged
    and skipped; every other account is disabled."""
    database = helper_database()
    sealer = SecretSealer(helper_settings().data_key)
    UserStore(database, sealer, BlindIndex(helper_settings().data_key)).owned("boss@example.com", "boss")
    database.executed(
        "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)", (b"s", "hash-ada", "Ada")
    )
    DocumentStore(database, sealer).created(1, "notes.txt", MediaType.TEXT, b"old", ProcessingStatus.UNSUPPORTED)
    report = helper_report_pdf()
    publications = [
        helper_publication("report.pdf", report),
        helper_publication("gone.pdf", b"%PDF-1.7 gone"),
        helper_publication("changed.pdf", b"%PDF-1.7 as listed"),
    ]
    irs = helper_irs({"report.pdf": report, "changed.pdf": b"%PDF-1.7 revised"})
    tested = AppFactory()
    fake = FakeSMTP()
    with (
        caplog.at_level(logging.WARNING, logger="rag_my_docs"),
        fake.installed(),
        FakeSentenceTransformer().installed(),
        irs.installed(),
        TestClient(tested.built(helper_settings(), publications)) as client,
    ):
        helper_owner_in(client, fake, 1)
        helper_settled(client, 2)
        listed = client.get("/api/documents")
        downloaded = client.get("/api/documents/2/content")
        stale = client.get("/api/documents/1/content")
        stored = database.row("SELECT name_sealed, content_sealed FROM documents WHERE id = %s", (2,))
        accounts = database.rows("SELECT id, disabled_at IS NOT NULL AS is_disabled FROM users ORDER BY id", ())
    result = [
        (listed.status_code, helper_unstamped(listed.json())),
        (
            downloaded.status_code,
            downloaded.content == report,
            downloaded.headers["content-type"],
            downloaded.headers["content-disposition"],
            downloaded.headers["cache-control"],
        ),
        (stale.status_code, stale.json()),
        accounts,
    ]
    expected = [
        (
            200,
            {
                "documents": [
                    {
                        "document_id": 2,
                        "user_id": 1,
                        "name": "report.pdf",
                        "media_type": "application/pdf",
                        "size_bytes": len(report),
                        "created_at": "<time>",
                        "processing_status": "completed",
                        "index_status": "indexed",
                        "chunk_count": 3,
                        "indexed_count": 3,
                    }
                ]
            },
        ),
        (
            200,
            True,
            "application/pdf",
            "attachment; filename=\"report.pdf\"; filename*=UTF-8''report.pdf",
            "private, no-store",
        ),
        (404, {"detail": "This document does not exist."}),
        [{"id": 1, "is_disabled": False}, {"id": 2, "is_disabled": True}],
    ]
    assert result == expected
    exp_requests = [
        ("https://www.irs.gov/pub/irs-pdf/report.pdf", "rag_my_docs corpus loader", 60.0),
        ("https://www.irs.gov/pub/irs-pdf/gone.pdf", "rag_my_docs corpus loader", 60.0),
        ("https://www.irs.gov/pub/irs-pdf/changed.pdf", "rag_my_docs corpus loader", 60.0),
    ]
    assert irs.requests == exp_requests
    exp_messages = [
        "[USERS] 1 other accounts disabled",
        "[CORPUS] https://www.irs.gov/pub/irs-pdf/gone.pdf not loaded: The publication could not be downloaded.",
        "[CORPUS] https://www.irs.gov/pub/irs-pdf/changed.pdf not loaded: its checksum changed, 16 bytes",
    ]
    assert caplog.messages == exp_messages
    # Sealed at rest: only the key opens the stored bytes back into the name and the file.
    assert stored is not None
    exp_stored = ("report.pdf", True)
    assert (
        sealer.opened(bytes(stored["name_sealed"])),
        sealer.opened_bytes(bytes(stored["content_sealed"])) == report,
    ) == exp_stored


@pytest.mark.database
def test_created__a_publication_no_longer_listed_is_deleted_with_its_sections_and_chunks() -> None:
    database = helper_database()
    tested = AppFactory()
    publications = [helper_publication("report.pdf", helper_report_pdf())]
    with (
        FakeSentenceTransformer().installed(),
        helper_irs({"report.pdf": helper_report_pdf()}).installed(),
        TestClient(tested.built(helper_settings(), publications)) as client,
    ):
        fake = FakeSMTP()
        with fake.installed():
            helper_owner_in(client, fake, 1)
        helper_settled(client, 1)
        before = helper_row_counts(database)
    waiter = threading.Event()
    after: dict[str, int] = {}
    with FakeSentenceTransformer().installed(), TestClient(tested.built(helper_settings(), [])):
        for _ in range(200):
            after = helper_row_counts(database)
            if after["sections"] == 0:
                break
            waiter.wait(0.05)
    documents = database.rows("SELECT id FROM documents", ())
    result = [before, after, documents]
    expected = [{"sections": 3, "chunks": 3}, {"sections": 0, "chunks": 0}, []]
    assert result == expected


@pytest.mark.database
def test_created__a_document_that_does_not_exist_is_not_found() -> None:
    helper_database()
    tested = AppFactory()
    fake = FakeSMTP()
    with (
        fake.installed(),
        FakeSentenceTransformer().installed(),
        TestClient(tested.built(helper_settings(), [])) as client,
    ):
        helper_owner_in(client, fake, 1)
        processing = client.get("/api/documents/99/processing")
        extraction = client.get("/api/documents/99/extraction")
        reprocessed = client.post("/api/documents/99/processing", headers=helper_origin())
        downloaded = client.get("/api/documents/99/content")
    not_found = {"detail": "This document does not exist."}
    result = [
        (processing.status_code, processing.json()),
        (extraction.status_code, extraction.json()),
        (reprocessed.status_code, reprocessed.json()),
        (downloaded.status_code, downloaded.json()),
    ]
    expected = [(404, not_found), (404, not_found), (404, not_found), (404, not_found)]
    assert result == expected


SYSTEM = (
    "You answer questions about U.S. federal income tax for tax year 2025.\n"
    "Use only the numbered passages below, taken from IRS publications. Use nothing you know from elsewhere.\n"
    "If the passages do not contain the answer, say so plainly and set found to false.\n"
    "Answer briefly, in the language of the question, with the exact figures and words of the passages.\n"
    'Reply with JSON only: {"found": true or false, "answer": "...", "passages": [the numbers of the passages\n'
    "the answer comes from]}."
)
# Bag-of-words embeddings: the question repeats the Results passage's words,
# so it is close enough to it (0.746) to reach the model, and Intro's (0.458) comes second.
QUESTION = "In the report results, sales grew in both regions: which grew faster, the north or the south?"


def helper_chat_body(model: str, system: str, user: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def helper_passage(rank: int, chunk: int, section_path: str, page: int, text: str) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": chunk,
        "chunk_position": chunk - 1,
        "document_id": 1,
        "document_name": "report.pdf",
        "section_id": chunk,
        "section_path": section_path,
        "start_page": page,
        "end_page": page,
        "text": text,
        "prefix_chars": 0,
        "similarity": "<similarity>",
    }


@pytest.mark.database
def test_created__answer_and_evaluation_flow() -> None:
    """The owner lists a local endpoint's models, adds it in Settings, sees
    its light and makes it answer; the corpus is extracted and indexed; a
    question retrieves its passages and the model answers from them, citing
    one; a question far from every passage is answered without the model;
    a test question runs through the same flow. A reprocess keeps the
    embeddings of unchanged chunks, a rebuild embeds them all again."""
    helper_database()
    tested = AppFactory()
    fake = FakeSMTP()
    reply = '{"found": true, "answer": "The north grew faster.", "passages": [1]}'
    models = b'{"object": "list", "data": [{"id": "good"}, {"id": "other"}]}'
    chat = FakeOpenAiChat([models, models, reply])
    connection = {
        "name": "Local",
        "kind": "openai",
        "base_url": "https://llm.example.com/v1/",
        "api_key": "the-local-key-AbCd",
        "model": "good",
    }
    publications = [helper_publication("report.pdf", helper_report_pdf())]
    with (
        fake.installed(),
        FakeSentenceTransformer().installed(),
        chat.installed(),
        helper_irs({"report.pdf": helper_report_pdf()}).installed(),
        TestClient(tested.built(helper_settings(), publications)) as client,
    ):
        helper_owner_in(client, fake, 1)
        offered = client.post(
            "/api/llm/models",
            json={"kind": "openai", "base_url": "https://llm.example.com/v1/", "api_key": "the-local-key-AbCd"},
            headers=helper_origin(),
        )
        created = client.post("/api/llm/connections", json=connection, headers=helper_origin())
        light = client.get("/api/llm/connections/1/health")
        routed = client.put("/api/llm/routing", json={"answers": 1}, headers=helper_origin())
        listed_connections = client.get("/api/llm/connections")
        indexed = helper_settled(client, 1)
        answered = client.post("/api/answers", json={"question": f" {QUESTION} ", "top_k": 2}, headers=helper_origin())
        not_covered = client.post("/api/answers", json={"question": "How do I bake bread?"}, headers=helper_origin())
        empty = client.post("/api/answers", json={"question": "   "}, headers=helper_origin())
        cases = client.get("/api/evaluation/cases")
        run = client.post("/api/evaluation/run", json={"index": 0, "top_k": 2}, headers=helper_origin())
        client.post("/api/documents/1/processing", headers=helper_origin())
        reprocessed = helper_settled(client, 1)
        rebuilt = client.post("/api/documents/1/index", json={"rebuild": True}, headers=helper_origin())
        after_rebuild = helper_settled(client, 1)
        all_queued = client.post("/api/index/all", headers=helper_origin())
    exp_connection = {
        "connection_id": 1,
        "name": "Local",
        "kind": "openai",
        "base_url": "https://llm.example.com/v1",
        "has_key": True,
        "key_hint": "…AbCd",
        "model": "good",
        "disable_thinking": True,
    }
    settings = [
        (offered.status_code, offered.json()),
        (created.status_code, created.json()),
        (light.status_code, {name: value for name, value in light.json().items() if name != "elapsed_ms"}),
        (routed.status_code, routed.json()),
        (listed_connections.status_code, listed_connections.json()),
    ]
    exp_settings = [
        (200, {"models": ["good", "other"]}),
        (200, exp_connection),
        (200, {"connection_id": 1, "is_ok": True, "detail": "Connected: the key is accepted and the model is there."}),
        (200, {"ok": True}),
        (200, {"connections": [exp_connection], "routing": {"answers": 1}}),
    ]
    assert settings == exp_settings
    prompt = f"[1] report.pdf › 2. Results, page 2\n{RESULTS}\n\n[2] report.pdf › 1. Introduction, page 1\n{INTRO}\n\nQuestion: {QUESTION}"
    exp_answer = {
        "question": QUESTION,
        "passages": [
            helper_passage(1, 3, "2. Results", 2, RESULTS),
            helper_passage(2, 1, "1. Introduction", 1, INTRO),
        ],
        "system": SYSTEM,
        "prompt": prompt,
        "answer": {
            "is_found": True,
            "text": "The north grew faster.",
            "citations": [{"passage": 1, "document": "report.pdf", "page": 2}],
        },
        "model": "good",
        "embedding_ms": "<ms>",
        "search_ms": "<ms>",
        "answer_ms": "<ms>",
    }
    not_covered_answer = {
        "is_found": False,
        "text": "The publications do not cover this question: no passage is close enough to it.",
        "citations": [],
    }
    result = [
        indexed["document"]["index_report"]["embedded"],
        (answered.status_code, helper_unscored(answered.json())),
        (
            not_covered.status_code,
            not_covered.json()["answer"],
            not_covered.json()["model"],
            not_covered.json()["prompt"],
            not_covered.json()["answer_ms"],
            len(not_covered.json()["passages"]),
        ),
        (empty.status_code, empty.json()),
        (
            cases.status_code,
            hashlib.md5(json.dumps(cases.json(), ensure_ascii=False).encode()).hexdigest(),
        ),
        (
            run.status_code,
            run.json()["missing_documents"],
            run.json()["missing_facts"],
            run.json()["is_retrieval_right"],
            run.json()["is_answer_right"],
            run.json()["result"]["answer"],
        ),
        helper_untimed(reprocessed)["document"]["index_report"],
        (rebuilt.status_code, rebuilt.json()),
        helper_untimed(after_rebuild)["document"]["index_report"],
        (all_queued.status_code, all_queued.json()),
    ]
    expected = [
        3,
        (200, exp_answer),
        # The default number of passages, three here: the report has three.
        (200, not_covered_answer, "", "", 0.0, 3),
        (400, {"detail": "Type a question to answer."}),
        # The test set as listed: a digest of the exact JSON (test_evaluation_constants holds the cases).
        (200, "13a8a9a9f26ee08450eb68612ddb63d4"),
        # The first test question, against this report: not covered, so no model call.
        (
            200,
            [
                "Publication 501 — Dependents, Standard Deduction, and Filing Information",
                "Publication 17 — Your Federal Income Tax",
            ],
            ["15,750"],
            False,
            False,
            not_covered_answer,
        ),
        # The same text after a reprocess: every embedding is kept.
        {"chunks": 3, "embedded": 0, "reused": 3, "batches": 0, "duration_ms": "<ms>"},
        (200, {"ok": True}),
        {"chunks": 3, "embedded": 3, "reused": 0, "batches": 2, "duration_ms": "<ms>"},
        (200, {"queued": 0}),
    ]
    assert result == expected
    exp_similar = [True, True]
    similarities = [passage["similarity"] for passage in answered.json()["passages"]]
    assert [similarities[0] >= 0.66, similarities[0] > similarities[1]] == exp_similar
    # What left the server: the model list twice, then the passages and the question, once.
    url = "https://llm.example.com/v1"
    light_request = (f"{url}/models", "GET", "Bearer the-local-key-AbCd", {}, 8.0)
    answer_request = (
        f"{url}/chat/completions",
        "POST",
        "Bearer the-local-key-AbCd",
        helper_chat_body("good", SYSTEM, prompt, 1500),
        120.0,
    )
    exp_requests = [light_request, light_request, answer_request]
    assert chat.requests == exp_requests


@pytest.mark.database
def test_created__sign_out_ends_the_session() -> None:
    """Signing out deletes the session row: the cookie, replayed, opens nothing."""
    helper_database()
    tested = AppFactory()
    fake = FakeSMTP()
    with (
        fake.installed(),
        FakeSentenceTransformer().installed(),
        TestClient(tested.built(helper_settings(), [])) as client,
    ):
        helper_owner_in(client, fake, 1)
        token = client.cookies.get("rag_my_docs_session")
        signed_out = client.post("/api/auth/sign-out", headers=helper_origin())
        after = client.get("/api/account")
        client.cookies.set("rag_my_docs_session", str(token))
        replayed = client.get("/api/account")
    result = [
        (signed_out.status_code, signed_out.json()),
        (after.status_code, after.json()),
        (replayed.status_code, replayed.json()),
    ]
    expected = [
        (200, {"ok": True}),
        (401, {"detail": "Please sign in."}),
        (401, {"detail": "Please sign in."}),
    ]
    assert result == expected


@pytest.mark.database
def test_created__pdf_processing_flow() -> None:
    """A publication of the corpus: the processing thread reads it, finds
    its sections and cuts its chunks; the inspection endpoints show all of it."""
    helper_database()
    tested = AppFactory()
    fake = FakeSMTP()
    publications = [helper_publication("report.pdf", helper_report_pdf())]
    with (
        fake.installed(),
        FakeSentenceTransformer().installed(),
        helper_irs({"report.pdf": helper_report_pdf()}).installed(),
        TestClient(tested.built(helper_settings(), publications)) as client,
    ):
        helper_owner_in(client, fake, 1)
        processing = helper_settled(client, 1)
        extraction = client.get("/api/documents/1/extraction")
        listed = client.get("/api/documents")
    result = [
        helper_untimed(helper_unstamped(processing)),
        (
            extraction.status_code,
            [(block["block_id"], block["page"], block["kind"], block["text"]) for block in extraction.json()["blocks"]],
        ),
        [document["processing_status"] for document in listed.json()["documents"]],
    ]
    expected = [
        {
            "document": {
                "document_id": 1,
                "name": "report.pdf",
                "media_type": "application/pdf",
                "status": "completed",
                "error": "",
                "page_count": 2,
                "chunking_version": DEFAULT_STRATEGY,
                "processed_at": "<time>",
                "report": {
                    "pages_without_text": [],
                    "pages_probably_scanned": [],
                    "blocks": 6,
                    "tables": 0,
                    "headers_and_footers": 0,
                    "headings": 3,
                    "bookmarks": 0,
                    "section_source": "numbered",
                },
                "index_status": "indexed",
                "index_model": "BAAI/bge-small-en-v1.5+context-v3",
                "index_error": "",
                "indexed_at": "<time>",
                # Three chunks, embedded two at a time: two batches.
                "index_report": {"chunks": 3, "embedded": 3, "reused": 0, "batches": 2, "duration_ms": "<ms>"},
                "chunk_count": 3,
                "indexed_count": 3,
                "current_strategy": DEFAULT_STRATEGY,
                "is_stale": False,
            },
            "sections": [
                {
                    "section_id": 1,
                    "parent_id": None,
                    "position": 0,
                    "level": 1,
                    "number": "1",
                    "title": "1. Introduction",
                    "source": "numbered",
                    "start_page": 1,
                    "end_page": 1,
                    "heading_block_id": 0,
                    "chunk_count": 1,
                },
                {
                    "section_id": 2,
                    "parent_id": 1,
                    "position": 1,
                    "level": 2,
                    "number": "1.1",
                    "title": "1.1 Scope",
                    "source": "numbered",
                    "start_page": 1,
                    "end_page": 1,
                    "heading_block_id": 2,
                    "chunk_count": 1,
                },
                {
                    "section_id": 3,
                    "parent_id": None,
                    "position": 2,
                    "level": 1,
                    "number": "2",
                    "title": "2. Results",
                    "source": "numbered",
                    "start_page": 2,
                    "end_page": 2,
                    "heading_block_id": 4,
                    "chunk_count": 1,
                },
            ],
            "chunks": [
                {
                    "chunk_id": 1,
                    "section_id": 1,
                    "position": 0,
                    "kind": "prose",
                    "text": INTRO,
                    "prefix_chars": 0,
                    "token_count": 24,
                    "start_page": 1,
                    "end_page": 1,
                    "spans": [{"block_id": 1, "start": 0, "end": 99}],
                    "strategy": DEFAULT_STRATEGY,
                    "embedding_model": "BAAI/bge-small-en-v1.5+context-v3",
                },
                {
                    "chunk_id": 2,
                    "section_id": 2,
                    "position": 1,
                    "kind": "prose",
                    "text": SCOPE,
                    "prefix_chars": 0,
                    "token_count": 16,
                    "start_page": 1,
                    "end_page": 1,
                    "spans": [{"block_id": 3, "start": 0, "end": 86}],
                    "strategy": DEFAULT_STRATEGY,
                    "embedding_model": "BAAI/bge-small-en-v1.5+context-v3",
                },
                {
                    "chunk_id": 3,
                    "section_id": 3,
                    "position": 2,
                    "kind": "prose",
                    "text": RESULTS,
                    "prefix_chars": 0,
                    "token_count": 18,
                    "start_page": 2,
                    "end_page": 2,
                    "spans": [{"block_id": 5, "start": 0, "end": 88}],
                    "strategy": DEFAULT_STRATEGY,
                    "embedding_model": "BAAI/bge-small-en-v1.5+context-v3",
                },
            ],
        },
        (
            200,
            [
                (0, 1, "heading", "1. Introduction"),
                (1, 1, "paragraph", INTRO),
                (2, 1, "heading", "1.1 Scope"),
                (3, 1, "paragraph", SCOPE),
                (4, 2, "heading", "2. Results"),
                (5, 2, "paragraph", RESULTS),
            ],
        ),
        ["completed"],
    ]
    assert result == expected


@pytest.mark.database
def test_created__reprocessing_gives_the_same_result_without_duplicates() -> None:
    database = helper_database()
    tested = AppFactory()
    fake = FakeSMTP()
    with (
        fake.installed(),
        FakeSentenceTransformer().installed(),
        helper_irs({"doc.pdf": helper_report_pdf()}).installed(),
        TestClient(tested.built(helper_settings(), [helper_publication("doc.pdf", helper_report_pdf())])) as client,
    ):
        helper_owner_in(client, fake, 1)
        first = helper_settled(client, 1)
        before = helper_row_counts(database)
        reprocessed = client.post("/api/documents/1/processing", headers=helper_origin())
        second = helper_settled(client, 1)
        after = helper_row_counts(database)
    result = [
        (reprocessed.status_code, reprocessed.json()),
        second["document"]["status"],
        helper_without_ids(second) == helper_without_ids(first),
        before,
        after,
    ]
    expected = [
        (200, {"ok": True}),
        "completed",
        True,
        {"sections": 3, "chunks": 3},
        {"sections": 3, "chunks": 3},
    ]
    assert result == expected


@pytest.mark.database
def test_created__reprocessing_after_a_configuration_change() -> None:
    """New chunk sizes make the stored chunks stale; reprocessing applies them."""
    helper_database()
    tested = AppFactory()
    fake = FakeSMTP()
    small = ChunkingSettings(target_tokens=8, max_tokens=12, min_tokens=2, overlap_sentences=0)
    publications = [helper_publication("report.pdf", helper_report_pdf())]
    with (
        fake.installed(),
        FakeSentenceTransformer().installed(),
        helper_irs({"report.pdf": helper_report_pdf()}).installed(),
        TestClient(tested.built(helper_settings(), publications)) as client,
    ):
        helper_owner_in(client, fake, 1)
        before = helper_settled(client, 1)
        session = str(client.cookies.get("rag_my_docs_session"))
    # Started again: the publication is stored already, so nothing is downloaded.
    with (
        FakeSentenceTransformer().installed(),
        TestClient(tested.built(helper_settings_with(small), publications)) as client,
    ):
        client.cookies.set("rag_my_docs_session", session)
        stale = client.get("/api/documents/1/processing").json()
        client.post("/api/documents/1/processing", headers=helper_origin())
        after = helper_settled(client, 1)
    result = [
        (before["document"]["chunking_version"], before["document"]["is_stale"], len(before["chunks"])),
        (stale["document"]["chunking_version"], stale["document"]["current_strategy"], stale["document"]["is_stale"]),
        (
            after["document"]["chunking_version"],
            after["document"]["is_stale"],
            [chunk["text"] for chunk in after["chunks"]],
            [section["chunk_count"] for section in after["sections"]],
        ),
    ]
    expected = [
        (DEFAULT_STRATEGY, False, 3),
        (DEFAULT_STRATEGY, SMALL_STRATEGY, True),
        (
            SMALL_STRATEGY,
            False,
            [
                "This report explains the yearly results.",
                "Dr. Smith led the work in 2026.",
                "Costs rose by 3.5 percent.",
                "The scope covers two regions.",
                "Each region reports its own sales figures every quarter.",
                "Sales grew in both regions.",
                "The north grew faster than the south.",
                "Margins stayed stable.",
            ],
            [3, 2, 3],
        ),
    ]
    assert result == expected


@pytest.mark.database
def test_created__a_malformed_pdf_fails() -> None:
    """Bytes that start like a PDF, with the checksum listed: the pipeline cannot read them."""
    helper_database()
    tested = AppFactory()
    fake = FakeSMTP()
    with (
        fake.installed(),
        FakeSentenceTransformer().installed(),
        helper_irs({"doc.pdf": b"%PDF-1.4 garbage"}).installed(),
        TestClient(tested.built(helper_settings(), [helper_publication("doc.pdf", b"%PDF-1.4 garbage")])) as client,
    ):
        helper_owner_in(client, fake, 1)
        processing = helper_settled(client, 1)
    result = (
        processing["document"]["status"],
        processing["document"]["error"],
        processing["document"]["page_count"],
        processing["sections"],
        processing["chunks"],
    )
    expected = ("failed", "This PDF could not be read: the file is damaged or is not a PDF.", None, [], [])
    assert result == expected


@pytest.mark.database
def test_created__a_scanned_pdf_has_no_text() -> None:
    helper_database()
    tested = AppFactory()
    fake = FakeSMTP()
    scanned = PdfFactory().page().image().page().image().built()
    with (
        fake.installed(),
        FakeSentenceTransformer().installed(),
        helper_irs({"doc.pdf": scanned}).installed(),
        TestClient(tested.built(helper_settings(), [helper_publication("doc.pdf", scanned)])) as client,
    ):
        helper_owner_in(client, fake, 1)
        processing = helper_settled(client, 1)
    result = (
        processing["document"]["status"],
        processing["document"]["error"],
        processing["document"]["report"],
        processing["chunks"],
    )
    expected = (
        "no_text",
        "No page of this PDF has extractable text; it is probably scanned (OCR is not supported yet).",
        {
            "pages_without_text": [1, 2],
            "pages_probably_scanned": [1, 2],
            "blocks": 0,
            "tables": 0,
            "headers_and_footers": 0,
            "headings": 0,
            "bookmarks": 0,
            "section_source": "",
        },
        [],
    )
    assert result == expected


@pytest.mark.database
def test_created__answers_need_a_model() -> None:
    helper_database()
    tested = AppFactory()
    fake = FakeSMTP()
    chat = FakeOpenAiChat([])
    with (
        fake.installed(),
        FakeSentenceTransformer().installed(),
        chat.installed(),
        TestClient(tested.built(helper_settings(), [])) as client,
    ):
        helper_owner_in(client, fake, 1)
        answered = client.post("/api/answers", json={"question": "Who won?"}, headers=helper_origin())
    result = (answered.status_code, answered.json())
    expected = (503, {"detail": "Answers need a language model: choose one in Settings › Language models."})
    assert result == expected
    exp_requests: list[tuple[Any, ...]] = []
    assert chat.requests == exp_requests


@pytest.mark.database
def test_created__documents_left_pending_are_processed_at_startup() -> None:
    """A document still waiting when the previous process stopped is queued
    again when the next one starts: its status row is the queue. Its owner
    is not the one in the settings: processing does not ask whose it is."""
    database = helper_database()
    database.executed(
        "INSERT INTO users (email_sealed, email_hash, display_name) VALUES (%s, %s, %s)", (b"s", "hash-ada", "Ada")
    )
    sealer = SecretSealer(helper_settings().data_key)
    DocumentStore(database, sealer).created(1, "left.pdf", MediaType.PDF, helper_report_pdf(), ProcessingStatus.PENDING)
    tested = AppFactory()
    waiter = threading.Event()
    statuses: list[dict[str, Any]] = []
    with FakeSentenceTransformer().installed(), TestClient(tested.built(helper_settings(), [])):
        for _ in range(200):
            statuses = database.rows("SELECT processing_status FROM documents WHERE id = 1", ())
            if statuses[0]["processing_status"] not in ("pending", "processing"):
                break
            waiter.wait(0.05)
    result = (statuses, helper_row_counts(database))
    expected = ([{"processing_status": "completed"}], {"sections": 3, "chunks": 3})
    assert result == expected


def helper_owner() -> UserAccount:
    return UserAccount(user_id=1, email="boss@example.com", display_name="boss", is_disabled=False)


def test_lifespan(caplog: pytest.LogCaptureFixture) -> None:
    """Start: migrate, set up the owner, start the job threads, queue the
    corpus for download, then the documents left waiting for extraction and
    for indexing. Stop: the threads drain before the database closes."""
    tested = AppFactory
    settings = helper_settings()
    database = Database("postgresql://u:p@h/d")
    order: list[str] = []
    with (
        patch.object(Database, "migrated", side_effect=[(1,)]) as mock_migrated,
        patch.object(OwnerCommand, "owner", side_effect=[helper_owner()]) as mock_owner,
        patch.object(JobQueue, "started", side_effect=lambda: order.append("started")) as mock_started,
        patch.object(CorpusCommand, "queued", side_effect=lambda owner: order.append("corpus")) as mock_corpus,
        patch.object(ProcessingCommand, "resumed", side_effect=lambda: order.append("resumed")) as mock_resumed,
        patch.object(IndexCommand, "resumed", side_effect=lambda: order.append("indexing")) as mock_indexing,
        patch.object(JobQueue, "stopped", side_effect=lambda: order.append("stopped")) as mock_stopped,
        patch.object(Database, "closed", side_effect=lambda: order.append("database")) as mock_closed,
        caplog.at_level(logging.INFO, logger="rag_my_docs"),
    ):
        owners = OwnerCommand.__new__(OwnerCommand)
        corpus = CorpusCommand.__new__(CorpusCommand)
        processing = ProcessingCommand.__new__(ProcessingCommand)
        indexing = IndexCommand.__new__(IndexCommand)
        queues = [JobQueue.__new__(JobQueue), JobQueue.__new__(JobQueue)]
        lifespan = tested.lifespan(settings, database, owners, corpus, processing, indexing, queues)

        async def scenario() -> None:
            async with lifespan(FastAPI()):
                order.append("serving")

        asyncio.run(scenario())
    result = order
    expected = ["started", "started", "corpus", "resumed", "indexing", "serving", "stopped", "stopped", "database"]
    assert result == expected
    exp_calls = [call()]
    assert mock_migrated.mock_calls == exp_calls
    exp_calls = [call("boss@example.com")]
    assert mock_owner.mock_calls == exp_calls
    exp_calls = [call(), call()]
    assert mock_started.mock_calls == exp_calls
    exp_calls = [call(helper_owner())]
    assert mock_corpus.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_resumed.mock_calls == exp_calls
    assert mock_indexing.mock_calls == exp_calls
    exp_calls = [call(), call()]
    assert mock_stopped.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_closed.mock_calls == exp_calls
    exp_messages = ["[DATABASE] migrations applied: (1,)"]
    assert [record.getMessage() for record in caplog.records] == exp_messages


def test_lifespan__nothing_to_migrate_and_no_owner(caplog: pytest.LogCaptureFixture) -> None:
    """Silent when the schema is current; without an owner no corpus is loaded."""
    tested = AppFactory
    database = Database("postgresql://u:p@h/d")
    with (
        patch.object(Database, "migrated", side_effect=[()]) as mock_migrated,
        patch.object(OwnerCommand, "owner", side_effect=[None]) as mock_owner,
        patch.object(JobQueue, "started", side_effect=[None]) as mock_started,
        patch.object(CorpusCommand, "queued", side_effect=[]) as mock_corpus,
        patch.object(ProcessingCommand, "resumed", side_effect=[None]) as mock_resumed,
        patch.object(IndexCommand, "resumed", side_effect=[None]) as mock_indexing,
        patch.object(JobQueue, "stopped", side_effect=[None]) as mock_stopped,
        patch.object(Database, "closed", side_effect=[None]) as mock_closed,
        caplog.at_level(logging.INFO, logger="rag_my_docs"),
    ):
        lifespan = tested.lifespan(
            helper_settings(),
            database,
            OwnerCommand.__new__(OwnerCommand),
            CorpusCommand.__new__(CorpusCommand),
            ProcessingCommand.__new__(ProcessingCommand),
            IndexCommand.__new__(IndexCommand),
            [JobQueue.__new__(JobQueue)],
        )

        async def scenario() -> None:
            async with lifespan(FastAPI()):
                pass

        asyncio.run(scenario())
    result = [record.getMessage() for record in caplog.records]
    expected: list[str] = []
    assert result == expected
    exp_calls = [call()]
    assert mock_migrated.mock_calls == exp_calls
    exp_calls = [call("boss@example.com")]
    assert mock_owner.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_started.mock_calls == exp_calls
    exp_calls = []
    assert mock_corpus.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_resumed.mock_calls == exp_calls
    assert mock_indexing.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_stopped.mock_calls == exp_calls
    exp_calls = [call()]
    assert mock_closed.mock_calls == exp_calls


def test_app_exception_handler() -> None:
    tested = AppFactory
    response = asyncio.run(tested.app_exception_handler(helper_request(), AppException(409, "Already there.")))
    result = (response.status_code, json.loads(response.body))
    expected = (409, {"detail": "Already there."})
    assert result == expected


def test_app_exception_handler__anything_else_is_raised_again() -> None:
    tested = AppFactory
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(tested.app_exception_handler(helper_request(), ValueError("not ours")))
    result = str(exc_info.value)
    expected = "not ours"
    assert result == expected
