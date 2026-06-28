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

from rag import dense_search, bm25_search, rag_pipeline


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


# ── Performance Report ────────────────────────────────────────────────────────

def generate_report(
    retrieval_metrics: Dict[str, float],
    quality_metrics: Dict[str, float],
    latency_stats: Dict[str, Dict[str, float]],
    output_path: str = "evaluation_report.html",
) -> str:
    """
    Generates an HTML report with tables and charts.
    Returns the output path.
    """
    # ── DataFrames ─────────────────────────────────────────────────────────
    ret_df = pd.DataFrame(
        [{"Metric": k, "Value": f"{v:.4f}"} for k, v in retrieval_metrics.items()]
    )
    qual_df = pd.DataFrame(
        [{"Metric": k, "Value": f"{v:.4f}"} for k, v in quality_metrics.items()]
    )
    lat_rows = []
    for comp, s in latency_stats.items():
        lat_rows.append({
            "Component": comp,
            "Mean (ms)": f"{s['mean_ms']:.1f}",
            "P50 (ms)":  f"{s['p50_ms']:.1f}",
            "P95 (ms)":  f"{s['p95_ms']:.1f}",
            "P99 (ms)":  f"{s['p99_ms']:.1f}",
        })
    lat_df = pd.DataFrame(lat_rows)

    # ── bar chart data ──────────────────────────────────────────────────────
    ret_keys   = list(retrieval_metrics.keys())
    ret_vals   = [round(v, 4) for v in retrieval_metrics.values()]
    qual_keys  = list(quality_metrics.keys())
    qual_vals  = [round(v, 4) for v in quality_metrics.values()]
    lat_labels = list(latency_stats.keys())
    lat_means  = [round(s["mean_ms"], 1) for s in latency_stats.values()]

    def table(df: pd.DataFrame, title: str) -> str:
        rows = "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
            for row in df.values
        )
        headers = "".join(f"<th>{c}</th>" for c in df.columns)
        return f"""
        <h2>{title}</h2>
        <table>
          <thead><tr>{headers}</tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAG Evaluation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; color: #222; }}
  h1   {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
  h2   {{ color: #34495e; margin-top: 40px; }}
  table{{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
  th   {{ background: #3498db; color: #fff; padding: 10px; text-align: left; }}
  td   {{ padding: 8px 10px; border-bottom: 1px solid #ddd; }}
  tr:hover {{ background: #f5f7fa; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; margin: 32px 0; }}
  canvas  {{ background: #fafafa; border-radius: 8px; padding: 8px; }}
</style>
</head>
<body>
<h1>RAG Evaluation Report</h1>

{table(ret_df,  "Retrieval Metrics")}
{table(qual_df, "Response Quality")}
{table(lat_df,  "Latency Benchmark")}

<div class="charts">
  <canvas id="retChart"></canvas>
  <canvas id="qualChart"></canvas>
  <canvas id="latChart"></canvas>
</div>

<script>
const BLUE = 'rgba(52,152,219,0.7)';
const GREEN = 'rgba(46,204,113,0.7)';
const ORANGE = 'rgba(230,126,34,0.7)';

new Chart(document.getElementById('retChart'), {{
  type: 'bar',
  data: {{
    labels: {ret_keys},
    datasets: [{{ label: 'Retrieval', data: {ret_vals}, backgroundColor: BLUE }}]
  }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Retrieval Metrics' }} }}, scales: {{ y: {{ min:0, max:1 }} }} }}
}});

new Chart(document.getElementById('qualChart'), {{
  type: 'bar',
  data: {{
    labels: {qual_keys},
    datasets: [{{ label: 'Quality', data: {qual_vals}, backgroundColor: GREEN }}]
  }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Response Quality' }} }}, scales: {{ y: {{ min:0, max:1 }} }} }}
}});

new Chart(document.getElementById('latChart'), {{
  type: 'bar',
  data: {{
    labels: {lat_labels},
    datasets: [{{ label: 'Mean ms', data: {lat_means}, backgroundColor: ORANGE }}]
  }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Latency (ms)' }} }} }}
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[evaluation] Report saved → {output_path}")
    return output_path
