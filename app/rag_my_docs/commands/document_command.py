from __future__ import annotations

from rag_my_docs.constants.document_constants import DocumentConstants
from rag_my_docs.libraries.document_store import DocumentStore
from rag_my_docs.structures.app_exception import AppException
from rag_my_docs.structures.document import Document
from rag_my_docs.structures.document_file import DocumentFile
from rag_my_docs.structures.user_account import UserAccount


class DocumentCommand:
    """The corpus, as the owner sees it: listed, and each publication handed
    back so a cited page can be checked. Documents are loaded by
    CorpusCommand, never uploaded."""

    def __init__(self, documents: DocumentStore) -> None:
        self._documents = documents

    def documents(self, user: UserAccount) -> list[Document]:
        return self._documents.of_user(user.user_id)

    def downloaded(self, user: UserAccount, document_id: int) -> DocumentFile:
        result = self._documents.file_of(user.user_id, document_id)
        if result is None:
            raise AppException(404, DocumentConstants.not_found)
        return result
