"""
rag.py — core retrieval + LLM pipeline.
Imports live globals from loader.py (populated at startup).
"""
import time
import json
import re
import unicodedata
from collections import defaultdict
from threading import Lock
from typing import List, Dict, Any, Tuple

import numpy as np
from groq import Groq
from nltk.tokenize import word_tokenize
from cachetools import TTLCache

import core.loader as loader
from core.config import settings

# ── caches ────────────────────────────────────────────────────────────────────
# dense_search : 1024 unique (query, top_k) pairs, expire after 1 hour
_dense_cache: TTLCache = TTLCache(maxsize=1024, ttl=3600)
_dense_lock = Lock()

# rewrite_query: 512 rewrites, expire after 30 minutes
_rewrite_cache: TTLCache = TTLCache(maxsize=512, ttl=1800)
_rewrite_lock = Lock()


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess_query(query: str) -> str:
    query = unicodedata.normalize("NFKC", query)
    query = query.lower()
    query = re.sub(r"http\S+", "", query)
    query = re.sub(r"\s+", " ", query)
    return query.strip()


# ── retrieval ──────────────────────────────────────────────────────────────────

def dense_search(query: str, top_k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    cache_key = (query, top_k)

    with _dense_lock:
        if cache_key in _dense_cache:
            return _dense_cache[cache_key]

    embedding = loader.embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    scores, ids = loader.faiss_index.search(np.array([embedding]), top_k)
    result = (ids[0], scores[0])

    with _dense_lock:
        _dense_cache[cache_key] = result

    return result


def bm25_search(query: str, top_k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    tokens = word_tokenize(query.lower())
    scores = loader.bm25.get_scores(tokens)
    ids = np.argsort(scores)[::-1][:top_k]
    return ids, scores[ids]


def retrieve(
    query: str,
    top_k_dense: int = settings.TOP_K_DENSE,
    top_k_bm25: int = settings.TOP_K_BM25,
    final_k: int = settings.FINAL_K,
    use_cross_encoder: bool = True,
    use_bm25: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Returns:
        results  — list of ranked chunks
        latency  — breakdown in ms
    """
    t0 = time.perf_counter()

    dense_ids, _ = dense_search(query, top_k_dense)
    candidate_ids = list(dense_ids)

    if use_bm25:
        bm25_ids, _ = bm25_search(query, top_k_bm25)
        candidate_ids = list(dict.fromkeys(candidate_ids + list(bm25_ids)))

    retrieval_ms = (time.perf_counter() - t0) * 1000

    if not candidate_ids:
        return [], {"retrieval_ms": retrieval_ms, "rerank_ms": 0.0}

    candidate_rows = loader.metadata.loc[candidate_ids].copy()

    # ── cross-encoder reranking ───────────────────────────────────────────────
    t1 = time.perf_counter()
    if use_cross_encoder:
        pairs = [(query, text) for text in candidate_rows["text_to_embed"]]
        rerank_scores = loader.cross_encoder.predict(pairs, batch_size=32)
        candidate_rows["rerank_score"] = rerank_scores
        candidate_rows = candidate_rows.sort_values("rerank_score", ascending=False)
    else:
        candidate_rows["rerank_score"] = 0.0
    rerank_ms = (time.perf_counter() - t1) * 1000

    # ── dedup + build results ─────────────────────────────────────────────────
    results, seen = [], set()
    for faiss_id, row in candidate_rows.iterrows():
        fp = " ".join(str(row["doc_chunk"]).split()[:50])
        if fp in seen:
            continue
        seen.add(fp)
        results.append({
            "faiss_id": int(faiss_id),
            "question": row.get("question", ""),
            "short_answer": row.get("short_answer", ""),
            "doc_chunk": row["doc_chunk"],
            "chunk_index": int(row.get("chunk_index", 0)),
            "language": row.get("language", "en"),
            "question_type": row.get("question_type", ""),
            "rerank_score": float(row["rerank_score"]),
        })
        if len(results) >= final_k:
            break

    return results, {"retrieval_ms": retrieval_ms, "rerank_ms": rerank_ms}


# ── LLM ───────────────────────────────────────────────────────────────────────

def call_llm(system: str, prompt: str, as_json: bool = False) -> str:
    client = Groq(api_key=settings.GROQ_API_KEY)
    kwargs: Dict[str, Any] = dict(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    )
    if as_json:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs).choices[0].message.content


def rewrite_query(query: str, history: list) -> str:
    """Rewrite to standalone question when there is conversation history."""
    if not history:
        return query

    # cache key = query + last 2 history turns (enough context, not too specific)
    history_key = str(history[-2:])
    cache_key = (query, history_key)

    with _rewrite_lock:
        if cache_key in _rewrite_cache:
            return _rewrite_cache[cache_key]

    prompt = f"""Rewrite the last question into a self-contained standalone question.

Conversation:
{history}

Last Question:
{query}

Output JSON only: {{"standalone_question": "..."}}"""
    raw = call_llm("You rewrite conversational questions into standalone questions.", prompt, as_json=True)
    rewritten = json.loads(raw)["standalone_question"].strip()

    with _rewrite_lock:
        _rewrite_cache[cache_key] = rewritten

    return rewritten


def build_prompt(history: list, context: str, query: str) -> Tuple[str, str]:
    system = (
        "You are a helpful assistant. "
        "Answer using ONLY the provided context. "
        "If the answer is not in the context, say: 'I don't know.'"
    )
    prompt = f"""Answer ONLY using the context below.

Conversation history:
{history}

Context:
{context}

Question:
{query}"""
    return system, prompt


# ── confidence ────────────────────────────────────────────────────────────────

def validate_answer(answer: str, context: str) -> float:
    a_emb = loader.embedding_model.encode(
        answer, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    c_emb = loader.embedding_model.encode(
        context, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    return float(np.dot(a_emb, c_emb))


# ── conversation memory ────────────────────────────────────────────────────────

class ConversationMemory:
    def __init__(self, max_history: int = settings.MAX_HISTORY):
        self.max_history = max_history
        self._sessions: Dict[str, list] = defaultdict(list)

    def add(self, session_id: str, role: str, content: str):
        self._sessions[session_id].append({"role": role, "content": content})
        if len(self._sessions[session_id]) > self.max_history:
            self._sessions[session_id] = self._sessions[session_id][-self.max_history:]

    def get(self, session_id: str) -> list:
        return self._sessions[session_id]

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)


memory = ConversationMemory()


# ── cache stats (for /health) ─────────────────────────────────────────────────

def cache_stats() -> Dict[str, Any]:
    return {
        "dense_cache":   {"size": len(_dense_cache),   "maxsize": _dense_cache.maxsize,   "ttl": _dense_cache.ttl},
        "rewrite_cache": {"size": len(_rewrite_cache), "maxsize": _rewrite_cache.maxsize, "ttl": _rewrite_cache.ttl},
    }


# ── main pipeline ──────────────────────────────────────────────────────────────

def rag_pipeline(
    session_id: str,
    question: str,
    use_bm25: bool = False,
    use_cross_encoder: bool = True,
) -> Dict[str, Any]:

    t_start = time.perf_counter()
    history = memory.get(session_id)

    question = preprocess_query(question)
    standalone = rewrite_query(question, history)

    docs, latency = retrieve(
        query=standalone,
        use_cross_encoder=use_cross_encoder,
        use_bm25=use_bm25,
    )

    if not docs:
        return {
            "question": standalone,
            "answer": "I couldn't find any relevant document.",
            "confidence": 0.0,
            "sources": [],
            "latency": latency,
        }

    context = "\n\n".join(d["doc_chunk"] for d in docs)
    system, prompt = build_prompt(history, context, standalone)

    t_llm = time.perf_counter()
    answer = call_llm(system, prompt)
    latency["llm_ms"] = (time.perf_counter() - t_llm) * 1000
    latency["total_ms"] = (time.perf_counter() - t_start) * 1000

    confidence = validate_answer(answer, context)

    if confidence < settings.CONFIDENCE_NONE:
        answer = "I don't know based on the provided context."
    elif confidence < settings.CONFIDENCE_LOW:
        answer += "\n\n Confidence is low."

    memory.add(session_id, "user", question)
    memory.add(session_id, "assistant", answer)

    return {
        "question": standalone,
        "answer": answer,
        "confidence": float(confidence),
        "sources": docs,
        "latency": latency,
    }