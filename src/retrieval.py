"""Hybrid retrieval logic for RAGauge."""

from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document


def build_hybrid_retriever(
    vector_store: Chroma,
    documents: list[Document],
    k: int = 4,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> EnsembleRetriever:
    """Build an ensemble retriever combining BM25 and vector similarity.

    Args:
        vector_store: Chroma vector store retriever source.
        documents: Documents/chunks to build BM25 index from.
        k: Number of top results to retrieve from each retriever.
        bm25_weight: Ensemble weight for BM25 retrieval.
        vector_weight: Ensemble weight for vector retrieval.

    Returns:
        Configured hybrid ensemble retriever.

    Raises:
        ValueError: If the document list is empty.
    """
    if not documents:
        raise ValueError("Cannot build retriever with an empty document set.")

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )
