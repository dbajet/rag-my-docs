from __future__ import annotations

import hashlib
import logging

from rag_my_docs.commands.processing_command import ProcessingCommand
from rag_my_docs.constants.constants import Constants
from rag_my_docs.constants.media_type import MediaType
from rag_my_docs.constants.processing_status import ProcessingStatus
from rag_my_docs.libraries.document_store import DocumentStore
from rag_my_docs.libraries.job_queue import JobQueue
from rag_my_docs.libraries.publication_box import PublicationBox
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.publication import Publication
from rag_my_docs.structures.user_account import UserAccount


class CorpusCommand:
    """The documents every question is asked of: a fixed list of
    publications, kept under the owner's account. At startup, on a job
    thread of its own, a stored document the list does not name is deleted,
    and each publication not stored yet is downloaded, checked against its
    pinned SHA-256, stored, and queued for processing — then embedding, as
    any document.

    A download that fails or does not match is logged and tried again at the
    next start; the others still load. Only a color that starts runs this,
    and blue/green starts one at a time, so two runs never race to store the
    same publication."""

    def __init__(
        self,
        publications: list[Publication],
        box: PublicationBox,
        documents: DocumentStore,
        processing: ProcessingCommand,
        jobs: JobQueue,
    ) -> None:
        self._publications = publications
        self._box = box
        self._documents = documents
        self._processing = processing
        self._jobs = jobs

    def queued(self, owner: UserAccount) -> None:
        self._jobs.queued(self.synced, (owner,))

    def synced(self, owner: UserAccount) -> None:
        stored = {document.name: document.document_id for document in self._documents.of_user(owner.user_id)}
        named = {publication.name for publication in self._publications}
        for name, document_id in stored.items():
            if name not in named:
                self._documents.deleted(owner.user_id, document_id)
        for publication in self._publications:
            if publication.name not in stored:
                self.loaded(owner, publication)

    def loaded(self, owner: UserAccount, publication: Publication) -> None:
        try:
            content = self._box.downloaded(publication.url)
        except AppException as error:
            logging.getLogger(Constants.logger_name).warning(
                "[CORPUS] %s not loaded: %s", publication.url, error.message
            )
            return
        if hashlib.sha256(content).hexdigest() != publication.sha256:
            logging.getLogger(Constants.logger_name).warning(
                "[CORPUS] %s not loaded: its checksum changed, %s bytes", publication.url, len(content)
            )
            return
        document = self._documents.created(
            owner.user_id, publication.name, MediaType.PDF, content, ProcessingStatus.PENDING
        )
        self._processing.queued(document.document_id)
