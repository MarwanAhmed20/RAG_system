"""
Loader — loads all heavy artifacts once at startup.
Call load_all() from main.py lifespan, then import the globals anywhere.
"""
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from core.config import settings

# ── globals ───────────────────────────────────────────────────────────────────
embedding_model: SentenceTransformer = None
cross_encoder: CrossEncoder = None
faiss_index: faiss.Index = None
metadata: pd.DataFrame = None
bm25: BM25Okapi = None


def load_all():
    global embedding_model, cross_encoder, faiss_index, metadata, bm25

    print("[loader] Loading embedding model...")
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    print("[loader] Loading cross-encoder...")
    cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL)

    print("[loader] Loading FAISS index...")
    faiss_index = faiss.read_index(settings.FAISS_INDEX_PATH)

    print("[loader] Loading metadata...")
    metadata = pd.read_parquet(settings.METADATA_PATH)

    print("[loader] Building BM25 index...")
    tokenized = [
        word_tokenize(t.lower())
        for t in metadata["text_to_embed"]
    ]
    bm25 = BM25Okapi(tokenized)

    print(
        f"[loader] Ready — "
        f"{faiss_index.ntotal} vectors | "
        f"{len(metadata)} rows"
    )


def health_status() -> dict:
    issues = []
    if embedding_model is None: issues.append("embedding model not loaded")
    if cross_encoder is None:   issues.append("cross encoder not loaded")
    if faiss_index is None:     issues.append("FAISS index not loaded")
    if metadata is None:        issues.append("metadata not loaded")
    if bm25 is None:            issues.append("BM25 index not loaded")

    if issues:
        return {"status": "degraded", "issues": issues}

    return {
        "status": "ok",
        "index_size": int(faiss_index.ntotal),
        "metadata_rows": int(len(metadata)),
        "embedding_model": settings.EMBEDDING_MODEL,
        "cross_encoder_model": settings.CROSS_ENCODER_MODEL,
    }
