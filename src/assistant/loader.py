import logging

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.assistant.pgvector import PGVectorUtils
from src.core.consts import LOADER_CLASS_MAP, PGVECTOR_DSN


logger = logging.getLogger(__name__)


class Loader:

    def __init__(
            self,
            documents_path,
            chunk_size: int = 2500,
            chunk_overlap: int = 250
    ) -> None:

        self.document_path = documents_path
        self.documents = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore_utils = PGVectorUtils(PGVECTOR_DSN)

    def __read_documents(self):
        # "./documents.yml"
        with open(self.document_path, "r") as file:
            self.documents = yaml.safe_load(file)

    def loader(self):
        if not self.documents:
            self.__read_documents()

        documents_data = []
        for document_types in self.documents:
            for document in self.documents[document_types]:
                cls = LOADER_CLASS_MAP[document_types]
                loader = cls(document)
                documents_data.extend(loader.load())

        return documents_data

    def split_data(self, documents_data):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        all_splits = text_splitter.split_documents(documents_data)
        return all_splits

    def commit(self):
        documents_data = self.loader()
        all_splits_data = self.split_data(documents_data)
        self.vectorstore_utils.load_to_pgvector(all_splits_data)
