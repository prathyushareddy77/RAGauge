"""Document ingestion and vector indexing for RAGauge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for document ingestion."""

    data_dir: Path = Path("data")
    persist_directory: Path = Path("chroma_db")
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str = "text-embedding-3-small"


class DocumentIngestor:
    """Loads source files, splits them into chunks, and indexes to ChromaDB."""

    def __init__(self, config: IngestionConfig) -> None:
        """Initialize the ingestor.

        Args:
            config: Ingestion configuration.
        """
        self.config = config

    def _load_documents(self) -> list[Document]:
        """Load all supported documents from the data directory.

        Returns:
            Loaded LangChain documents.

        Raises:
            FileNotFoundError: If the data directory does not exist.
            ValueError: If no supported documents are found.
        """
        if not self.config.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.config.data_dir.resolve()}"
            )

        supported_extensions = {".pdf", ".txt", ".md"}
        available_files = [
            path
            for path in self.config.data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in supported_extensions
        ]
        if not available_files:
            raise ValueError(
                "No supported files found in data/. Add at least one PDF, TXT, or MD file."
            )

        loaders: Sequence[DirectoryLoader] = (
            DirectoryLoader(
                str(self.config.data_dir),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                silent_errors=False,
            ),
            DirectoryLoader(
                str(self.config.data_dir),
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                silent_errors=False,
            ),
            DirectoryLoader(
                str(self.config.data_dir),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                silent_errors=False,
            ),
        )

        documents: list[Document] = []
        for loader in loaders:
            documents.extend(loader.load())
        return documents

    def ingest(self) -> tuple[Chroma, list[Document]]:
        """Load, split, and index documents.

        Returns:
            A tuple containing:
            - Chroma vector store
            - split chunk documents used for retrieval
        """
        raw_documents = self._load_documents()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        split_documents = splitter.split_documents(raw_documents)

        self.config.persist_directory.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=split_documents,
            embedding=OpenAIEmbeddings(model=self.config.embedding_model),
            persist_directory=str(self.config.persist_directory),
        )
        return vector_store, split_documents
