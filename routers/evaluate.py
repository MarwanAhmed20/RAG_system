from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

from evaluation import (
    evaluate_retrieval,
    evaluate_response_quality,
    benchmark_latency,
    generate_report,
)
from rag import dense_search

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


class RetrievalCase(BaseModel):
    query: str
    relevant_ids: List[int]


class QualityCase(BaseModel):
    query: str
    reference_answer: str


class EvaluateRequest(BaseModel):
    retrieval_cases: Optional[List[RetrievalCase]] = None
    quality_cases: Optional[List[QualityCase]] = None
    benchmark_queries: Optional[List[str]] = None
    k_values: List[int] = [1, 3, 5]
    benchmark_runs: int = 3
    generate_html_report: bool = False


@router.post("")
def evaluate(req: EvaluateRequest, background_tasks: BackgroundTasks):
    if not req.retrieval_cases and not req.quality_cases and not req.benchmark_queries:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: retrieval_cases, quality_cases, benchmark_queries.",
        )

    output: dict = {}
    retrieval_metrics: dict = {}
    quality_metrics: dict = {}
    latency_stats: dict = {}

    if req.retrieval_cases:
        retrieval_metrics = evaluate_retrieval(
            test_cases=[tc.model_dump() for tc in req.retrieval_cases],
            search_fn=dense_search,
            k_values=req.k_values,
        )
        output["retrieval"] = retrieval_metrics

    if req.quality_cases:
        quality_metrics = evaluate_response_quality(
            test_cases=[tc.model_dump() for tc in req.quality_cases],
        )
        output["response_quality"] = quality_metrics

    if req.benchmark_queries:
        latency_stats = benchmark_latency(
            queries=req.benchmark_queries,
            runs=req.benchmark_runs,
        )
        output["latency"] = latency_stats

    if req.generate_html_report and (retrieval_metrics or quality_metrics or latency_stats):
        background_tasks.add_task(
            generate_report,
            retrieval_metrics or {},
            quality_metrics or {},
            latency_stats or {},
        )
        output["report"] = "generating in background → evaluation_report.html"

    return output
