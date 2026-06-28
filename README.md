# RAG Q&A API

A production-ready Retrieval-Augmented Generation system built on FAISS, sentence-transformers, BM25, a cross-encoder re-ranker, and Groq LLM — served via FastAPI with SQLite persistence.

---

## Project Structure

```
rag_project/
├── config.py          # All settings (paths, model names, thresholds)
├── loader.py          # Loads model/index/BM25 once at startup
├── rag.py             # Core RAG logic: retrieval, LLM, conversation memory
├── evaluation.py      # Precision@K, Recall@K, BLEU, ROUGE, latency, HTML report
├── database.py        # SQLAlchemy engine + session
├── models/
│   ├── db_models.py   # Query / Response / Analytics ORM models
│   ├── faiss_index.bin
│   └── metadata.parquet
├── routers/
│   ├── qa.py          # POST /ask-question
│   ├── health.py      # GET  /health
│   └── evaluate.py    # POST /evaluate
├── main.py            # FastAPI app entry point
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Requirements

- Python 3.11+
- `models/faiss_index.bin` — built from your notebook
- `models/metadata.parquet` — must include `text_to_embed` column
- Groq API key → [console.groq.com](https://console.groq.com)

---

## Quickstart
### 0. download index and metadata

```bash
python download_index.py
```

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

### 2. Set environment variable

```bash
export GROQ_API_KEY="gsk_..."
```

### 3. Run

```bash
uvicorn main:app --reload
```

API is live at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## Docker



```bash
python download_index.py
```

### Build

```bash
docker build -t rag-api .
```

### Run

Create an environment file `.env` with required secrets (example keys below), then run:

```bash
sudo docker run -d --name rag-cont \
  --env-file .env \
  -p 8000:8000 \
  rag-api
```


### Persist the SQLite database across restarts

```bash
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  rag-api
```

---

## API Reference

### `POST /ask-question`

Main Q&A endpoint.

**Request**
```json
{
  "session_id": "user_123",
  "question": "What type of fertilisation takes place in humans?",
  "use_bm25": false,
  "use_cross_encoder": true
}
```

**Response**
```json
{
  "question": "What type of fertilisation takes place in humans?",
  "answer": "Human fertilization is the union of a human egg and sperm.",
  "confidence": 0.854,
  "sources": [
    {
      "faiss_id": 4659,
      "doc_chunk": "Human fertilization is the union...",
      "rerank_score": 7.17
    }
  ],
  "latency": {
    "retrieval_ms": 65.2,
    "rerank_ms": 98.4,
    "llm_ms": 310.1,
    "total_ms": 490.3
  }
}
```

---

### `DELETE /ask-question/session/{session_id}`

Clears conversation history for a session.

---

### `GET /health`

System health check.

**Response**
```json
{
  "status": "ok",
  "index_size": 86212,
  "metadata_rows": 86212,
  "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
  "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

---

### `POST /evaluate`

Runs evaluation and optionally generates an HTML report.

**Request**
```json
{
    "retrieval_cases": [
      { 
        "query": "What is fertilisation?", 
        "relevant_ids": [74508, 58981, 11561] 
      },
      { 
        "query": "Who played Mantis in Guardians of the Galaxy 2?", 
        "relevant_ids": [14735, 70319, 48347, 57866] 
      }
    ],
    "quality_cases": [
      { 
        "query": "Who played Mantis in Guardians of the Galaxy 2?", 
        "reference_answer": "Pom Klementieff played the role of Mantis in Guardians of the Galaxy Vol. 2" 
      },
      {
        "query": "What is fertilisation?",
        "reference_answer": "Fertilization is the union of a human egg and sperm, usually occurring in the ampulla of the fallopian tube"
      }
    ],
    "benchmark_queries": [
      "What type of fertilisation takes place in humans",
      "Who played Mantis in Guardians of the Galaxy 2"
    ],
    "k_values": [1, 3, 5],
    "benchmark_runs": 3,
  }
```

**Response**
```json
{
  "retrieval": {
    "precision@1": 0.77,
    "recall@5": 0.95,
    "mrr@5": 0.85
  },
  "response_quality": {
    "bleu": 0.54,
    "rouge1": 0.98,
    "rougeL": 0.98
  },
  "latency": {
    "dense":         { "mean_ms": 68.9, "p95_ms": 102.7 },
    "full_pipeline": { "mean_ms": 545.7, "p95_ms": 776.7 }
  },
  "report": "generating in background → evaluation_report.html"
}
```

---

## Configuration

All settings live in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `FAISS_INDEX_PATH` | `models/faiss_index.bin` | Path to FAISS index |
| `METADATA_PATH` | `models/metadata.parquet` | Path to metadata parquet |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | Sentence transformer |
| `CROSS_ENCODER_MODEL` | `ms-marco-MiniLM-L-6-v2` | Re-ranker |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM |
| `TOP_K_DENSE` | `20` | Dense retrieval candidates |
| `FINAL_K` | `5` | Results returned after re-ranking |
| `CONFIDENCE_NONE` | `0.30` | Below this → "I don't know" |
| `CONFIDENCE_LOW` | `0.65` | Below this → low confidence warning |
| `MAX_HISTORY` | `6` | Conversation turns to keep |
| `DB_URL` | `sqlite:///rag.db` | Database connection string |

---

## Database Schema

| Table | Columns |
|-------|---------|
| `queries` | id, session_id, raw_question, standalone_question, created_at |
| `responses` | id, query_id, answer, confidence, sources (JSON), created_at |
| `analytics` | id, query_id, retrieval_ms, rerank_ms, llm_ms, total_ms, num_sources, used_bm25, used_cross_encoder |

---

## Evaluation Results

Benchmarked on 100 samples from Natural Questions:

| Metric | Value |
|--------|-------|
| precision@1 | 0.77 |
| recall@5 | 0.95 |
| MRR@5 | 0.85 |
| BLEU | 0.54 |
| ROUGE-1 | 0.98 |
| ROUGE-L | 0.98 |
| Dense latency (mean) | ~69ms |
| Full pipeline (mean) | ~546ms |
