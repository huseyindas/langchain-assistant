import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.sitemap import SitemapLoader


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://0.0.0.0:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen:0.5b")

CHUNK_SIZE = os.getenv("CHUNK_SIZE", 2500)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 250)

LOADER_CLASS_MAP = {
    "sitemaps": SitemapLoader,
    "pdfs": PyPDFLoader,
}

PGVECTOR_DSN = os.getenv(
    "PGVECTOR_DSN",
    "postgresql+psycopg://langchain:langchain@0.0.0.0:5433/langchain"
)

DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "./documents.yml")
PROMPT_PATH = os.getenv("PROMPT_PATH", "./prompt.txt")
