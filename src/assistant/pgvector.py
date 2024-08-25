import logging

from langchain_postgres import PGVector

from src.assistant.embedding import CustomOllamaEmbeddings
from src.core.consts import PGVECTOR_DSN


logger = logging.getLogger(__name__)


class PGVectorUtils:

    def __init__(self, pgvector_dsn: str = PGVECTOR_DSN) -> None:
        self.pgvector_dsn = pgvector_dsn
        self.embedding = CustomOllamaEmbeddings()
        self.retriever_score_threshold = 0.5
        self.retriever_k = 2

    def get_vectorstore(self):
        vectorstore = PGVector(
            embeddings=self.embedding,
            connection=self.pgvector_dsn
            )
        return vectorstore

    def get_retriever(self):
        vectorstore = self.get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": self.retriever_score_threshold,
                "k": self.retriever_k
            }
        )
        return retriever

    def load_to_pgvector(self, all_splits_data):
        vectorstore = self.get_vectorstore()

        for document in all_splits_data:
            try:
                _ = vectorstore.from_documents(
                        documents=[document],
                        embedding=self.embedding,
                        connection=self.pgvector_dsn
                    )
                logging.info("A document added to vectorstore.")
            except Exception as ex:
                logger.error("Error on Loader.load_to_pgvector")
                logger.error(ex)