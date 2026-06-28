from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import random

import core.loader as loader
from core.evaluation import (
    evaluate_retrieval,
    evaluate_response_quality,
    benchmark_latency,
)
from core.rag import dense_search

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

    return output



class AutoEvaluateRequest(BaseModel):
    sample_size: int = 100
    k_values: List[int] = [1, 3, 5]
    benchmark_runs: int = 3
    generate_html_report: bool = False
    seed: int = 42


@router.post("/auto")
def evaluate_auto(req: AutoEvaluateRequest, background_tasks: BackgroundTasks):
    """
    Builds test cases automatically from metadata:
    - relevant_ids  = all chunk indices sharing the same question
    - reference_answer = short_answer from metadata
    """
    df = loader.metadata

    # sample unique questions that have a short_answer
    valid = df[df["short_answer"].notna() & (df["short_answer"].str.strip() != "")]
    unique_questions = valid["question"].unique().tolist()

    random.seed(req.seed)
    sampled = random.sample(
        unique_questions,
        min(req.sample_size, len(unique_questions))
    )

    retrieval_cases = []
    quality_cases = []

    for q in sampled:
        rows = df[df["question"] == q]

        # relevant_ids = all chunk indices for this question
        relevant_ids = rows.index.tolist()

        # reference_answer = most common short_answer for this question
        reference_answer = rows["short_answer"].mode()[0]

        retrieval_cases.append({
            "query": q,
            "relevant_ids": relevant_ids,
        })
        quality_cases.append({
            "query": q,
            "reference_answer": str(reference_answer),
        })

    # retrieval metrics
    retrieval_metrics = evaluate_retrieval(
        test_cases=retrieval_cases,
        search_fn=dense_search,
        k_values=req.k_values,
    )

    # response quality
    quality_metrics = evaluate_response_quality(
        test_cases=quality_cases,
    )

    # latency benchmark using the sampled questions as queries
    benchmark_queries = [tc["query"] for tc in retrieval_cases[:10]]
    latency_stats = benchmark_latency(
        queries=benchmark_queries,
        runs=req.benchmark_runs,
    )

    output = {
        "sample_size": len(sampled),
        "retrieval": retrieval_metrics,
        "response_quality": quality_metrics,
        "latency": latency_stats,
    }

    if req.generate_html_report:
        background_tasks.add_task(
            retrieval_metrics,
            quality_metrics,
            latency_stats,
        )
    return output