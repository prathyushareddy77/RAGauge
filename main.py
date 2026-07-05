"""Main orchestration entrypoint for the RAGauge evaluation pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.evaluation import EvaluationSample, run_evaluation
from src.ingestion import DocumentIngestor, IngestionConfig
from src.retrieval import build_hybrid_retriever


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration values for the full RAGauge pipeline."""

    data_dir: Path = Path("data")
    tests_file: Path = Path("tests/ground_truth.json")
    failed_queries_file: Path = Path("logs/failed_queries.json")
    chroma_dir: Path = Path("chroma_db")
    retrieval_k: int = 4
    chat_model: str = "gpt-4o-mini"


class Pipeline:
    """Coordinates ingestion, retrieval, and RAG evaluation."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize pipeline resources and validate environment."""
        load_dotenv()
        self.config = config or PipelineConfig()
        self._validate_environment()

        self.ingestor = DocumentIngestor(
            IngestionConfig(
                data_dir=self.config.data_dir,
                persist_directory=self.config.chroma_dir,
            )
        )
        self.vector_store = None
        self.documents: list[Document] = []
        self.retriever = None
        self.llm = ChatOpenAI(model=self.config.chat_model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    @staticmethod
    def _validate_environment() -> None:
        """Ensure required environment variables are present.

        Raises:
            ValueError: If OPENAI_API_KEY is missing.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is missing. Set it in your environment or .env file."
            )

    def setup(self) -> None:
        """Run ingestion and initialize the hybrid retriever."""
        self.vector_store, self.documents = self.ingestor.ingest()
        self.retriever = build_hybrid_retriever(
            vector_store=self.vector_store,
            documents=self.documents,
            k=self.config.retrieval_k,
        )

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant context documents for a query."""
        if self.retriever is None:
            raise RuntimeError("Pipeline is not initialized. Call setup() first.")
        return self.retriever.invoke(query)

    def answer_question(self, question: str) -> tuple[str, list[str]]:
        """Generate an answer from retrieved context.

        Args:
            question: User question to answer.

        Returns:
            Tuple of generated answer and retrieved context strings.
        """
        docs = self.retrieve(question)
        contexts = [doc.page_content for doc in docs]
        context_blob = "\n\n".join(contexts)

        prompt = (
            "You are a precise assistant. Answer the user question using only the provided context.\n"
            "If the context is insufficient, say so clearly.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context_blob}"
        )
        response = self.llm.invoke(prompt)
        return str(response.content), contexts

    def _load_ground_truth(self) -> list[dict[str, str]]:
        """Load and validate test records.

        Returns:
            Ground truth records with question and ground_truth keys.

        Raises:
            FileNotFoundError: If test file does not exist.
            ValueError: If JSON content is malformed.
        """
        if not self.config.tests_file.exists():
            raise FileNotFoundError(
                f"Ground truth file not found: {self.config.tests_file.resolve()}"
            )

        raw_text = self.config.tests_file.read_text(encoding="utf-8")
        records = json.loads(raw_text)
        if not isinstance(records, list):
            raise ValueError("ground_truth.json must contain a JSON list.")

        for record in records:
            if "question" not in record or "ground_truth" not in record:
                raise ValueError(
                    "Each test record must include 'question' and 'ground_truth'."
                )

        return records

    def evaluate(self) -> dict[str, float]:
        """Run end-to-end answer generation and Ragas evaluation."""
        samples: list[EvaluationSample] = []
        for record in self._load_ground_truth():
            answer, contexts = self.answer_question(record["question"])
            samples.append(
                EvaluationSample(
                    question=record["question"],
                    answer=answer,
                    ground_truth=record["ground_truth"],
                    contexts=contexts,
                )
            )

        return run_evaluation(
            samples=samples,
            failed_queries_path=self.config.failed_queries_file,
            llm=self.llm,
            embeddings=self.embeddings,
        )

    def run(self) -> dict[str, float]:
        """Execute the full pipeline."""
        self.setup()
        return self.evaluate()


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.setup()

    smoke_query = "Provide a summary of the available project documents."
    smoke_results = pipeline.retrieve(smoke_query)
    print(f"Smoke test status: OK. Retrieved {len(smoke_results)} document chunks.")

    metric_scores = pipeline.evaluate()
    print(f"Pipeline status: completed. Metrics: {metric_scores}")
