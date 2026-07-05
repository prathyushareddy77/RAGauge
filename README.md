# RAGauge

RAGauge is a modular, production-ready RAG evaluation pipeline that ingests documents, retrieves hybrid context, and evaluates answers with Ragas metrics.

## Setup Instructions

1. Create and activate a virtual environment:
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Configure environment variables in `.env`:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the full pipeline:

```powershell
python main.py
```

The pipeline will:
- ingest files from `data/`,
- build a hybrid retriever (BM25 + Chroma),
- evaluate against `tests/ground_truth.json`,
- print metric summary in the terminal,
- write low-faithfulness failures to `logs/failed_queries.json`.

## Data Format

- `data/`: include unstructured files in `.pdf`, `.txt`, or `.md` formats.
- `tests/ground_truth.json`: JSON list with:
  ```json
  [
    {
      "question": "What is ...?",
      "ground_truth": "Expected factual answer."
    }
  ]
  ```

## Troubleshooting

- **Missing API key**: If `OPENAI_API_KEY` is not set, startup raises a clear `ValueError`.
- **Dependency issues**: Re-run `pip install -r requirements.txt` in the active virtual environment.
