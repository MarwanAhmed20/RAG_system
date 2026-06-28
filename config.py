import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # ── Paths ────────────────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "models/faiss_index.bin"
    METADATA_PATH: str = "models/metadata.parquet"

    # ── Models ───────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Groq ─────────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K_DENSE: int = 20
    TOP_K_BM25: int = 20
    FINAL_K: int = 5
    BATCH_SIZE: int = 128

    # ── Confidence ────────────────────────────────────────────────────────────
    CONFIDENCE_NONE: float = 0.30
    CONFIDENCE_LOW: float = 0.65

    # ── Conversation ──────────────────────────────────────────────────────────
    MAX_HISTORY: int = 6

    # ── Database ──────────────────────────────────────────────────────────────
    DB_URL: str = "sqlite:///rag.db"


settings = Settings()
