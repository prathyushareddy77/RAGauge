"""Ragas-based evaluation logic for RAGauge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference
from tabulate import tabulate


class EvaluationSample(TypedDict):
    """One evaluation row for Ragas."""

    question: str
    answer: str
    ground_truth: str
    contexts: list[str]


def _validate_samples(samples: list[EvaluationSample]) -> None:
    """Validate evaluation samples before running metrics.

    Args:
        samples: Collection of evaluation samples.

    Raises:
        ValueError: If data is empty or malformed.
    """
    if not samples:
        raise ValueError("No evaluation samples were provided.")

    for sample in samples:
        missing = [key for key in ("question", "answer", "ground_truth", "contexts") if key not in sample]
        if missing:
            raise ValueError(f"Evaluation sample is missing required fields: {missing}")
        if not isinstance(sample["contexts"], list):
            raise ValueError("Sample 'contexts' must be a list of strings.")


def run_evaluation(
    samples: list[EvaluationSample],
    failed_queries_path: Path | str,
    llm: Any,
    embeddings: Any,
    faithfulness_threshold: float = 0.5,
) -> dict[str, float]:
    """Run Ragas metrics and persist failed-query diagnostics.

    Args:
        samples: Prepared question/answer/context/ground truth records.
        failed_queries_path: Output JSON path for low-faithfulness rows.
        llm: LangChain chat model used by Ragas metrics.
        embeddings: LangChain embeddings model for metrics that need embeddings.
        faithfulness_threshold: Threshold under which rows are logged.

    Returns:
        Aggregate metric scores keyed by metric name.
    """
    _validate_samples(samples)

    evaluation_dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": sample["question"],
                "response": sample["answer"],
                "reference": sample["ground_truth"],
                "retrieved_contexts": sample["contexts"],
            }
            for sample in samples
        ]
    )

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    faithfulness_metric = Faithfulness()
    context_precision_metric = LLMContextPrecisionWithReference()

    evaluation_result = evaluate(
        dataset=evaluation_dataset,
        metrics=[faithfulness_metric, context_precision_metric],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=True,
        show_progress=False,
    )
    result_dict = evaluation_result.to_dict()

    aggregate_scores = {
        "faithfulness": float(result_dict["faithfulness"]),
        "context_precision": float(
            result_dict.get(
                "context_precision",
                result_dict.get("llm_context_precision_with_reference"),
            )
        ),
    }

    table_rows = [[metric, f"{score:.4f}"] for metric, score in aggregate_scores.items()]
    print(tabulate(table_rows, headers=["Metric", "Score"], tablefmt="github"))

    failed_queries: list[dict[str, Any]] = []
    result_frame = evaluation_result.to_pandas()
    for _, row in result_frame.iterrows():
        row_faithfulness = float(row["faithfulness"])
        if row_faithfulness < faithfulness_threshold:
            failed_queries.append(
                {
                    "question": str(row["user_input"]),
                    "reason": f"Faithfulness below threshold {faithfulness_threshold}.",
                    "score": row_faithfulness,
                }
            )

    failed_path = Path(failed_queries_path)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text(json.dumps(failed_queries, indent=2), encoding="utf-8")

    return aggregate_scores
