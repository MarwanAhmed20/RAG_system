"""
evaluation.py — retrieval metrics, response quality, latency benchmarks,
and performance report generation.
"""
import time
from typing import List, Dict, Callable, Any

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rs

from core.rag import dense_search, bm25_search, rag_pipeline


# ── Retrieval Metrics ──────────────────────────────────────────────────────────

def precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def mrr_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    for rank, r in enumerate(retrieved[:k], 1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    test_cases: List[Dict],
    search_fn: Callable = dense_search,
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """
    test_cases: [{"query": str, "relevant_ids": List[int]}]
    """
    results = {
        f"{m}@{k}": []
        for k in k_values
        for m in ["precision", "recall", "mrr"]
    }

    for tc in test_cases:
        ids, _ = search_fn(tc["query"], top_k=max(k_values))
        retrieved = ids.tolist()
        relevant = tc["relevant_ids"]
        for k in k_values:
            results[f"precision@{k}"].append(precision_at_k(retrieved, relevant, k))
            results[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            results[f"mrr@{k}"].append(mrr_at_k(retrieved, relevant, k))

    return {m: float(np.mean(v)) for m, v in results.items()}


# ── Response Quality ───────────────────────────────────────────────────────────

def compute_bleu(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    return float(sentence_bleu(
        [ref], hyp,
        smoothing_function=SmoothingFunction().method1,
    ))


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, hypothesis)
    return {k: float(v.fmeasure) for k, v in s.items()}


def evaluate_response_quality(
    test_cases: List[Dict],
    session_id: str = "eval_session",
) -> Dict[str, float]:
    """
    test_cases: [{"query": str, "reference_answer": str}]
    """
    bleu, r1, r2, rl = [], [], [], []

    for tc in test_cases:
        result = rag_pipeline(session_id, tc["query"])
        hyp = result.get("answer", "")
        ref = tc["reference_answer"]

        bleu.append(compute_bleu(ref, hyp))
        rouge = compute_rouge(ref, hyp)
        r1.append(rouge["rouge1"])
        r2.append(rouge["rouge2"])
        rl.append(rouge["rougeL"])

    return {
        "bleu":   float(np.mean(bleu)),
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rl)),
    }


# ── Latency Benchmark ─────────────────────────────────────────────────────────

def benchmark_latency(
    queries: List[str],
    runs: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmarks dense, bm25, hybrid, and full pipeline latency.
    Returns per-component stats (mean, p50, p95, p99) in ms.
    """
    components = {
        "dense": lambda q: dense_search(q, top_k=20),
        "bm25":  lambda q: bm25_search(q, top_k=20),
        "hybrid": lambda q: (
            dense_search(q, top_k=20),
            bm25_search(q, top_k=20),
        ),
        "full_pipeline": lambda q: rag_pipeline("bench_session", q),
    }

    timings: Dict[str, List[float]] = {k: [] for k in components}

    for _ in range(runs):
        for q in queries:
            for name, fn in components.items():
                t0 = time.perf_counter()
                fn(q)
                timings[name].append((time.perf_counter() - t0) * 1000)

    stats = {}
    for name, times in timings.items():
        arr = np.array(times)
        stats[name] = {
            "mean_ms":  float(arr.mean()),
            "p50_ms":   float(np.percentile(arr, 50)),
            "p95_ms":   float(np.percentile(arr, 95)),
            "p99_ms":   float(np.percentile(arr, 99)),
        }
    return stats
