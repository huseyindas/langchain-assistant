import logging
from typing import List

from ollama import Client
from langchain_ollama import OllamaEmbeddings

from src.core.consts import OLLAMA_HOST


logger = logging.getLogger(__name__)


class CustomOllamaEmbeddings(OllamaEmbeddings):

    model: str = "nomic-embed-text"
    host: str = OLLAMA_HOST

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        client = Client(host=self.host)
        embedded_docs = client.embed(self.model, texts)["embeddings"]
        return embedded_docs
