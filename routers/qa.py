from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from database import get_db
from models.db_models import Query, Response, Analytics
from rag import rag_pipeline, memory

router = APIRouter(prefix="/ask-question", tags=["Q&A"])


class AskRequest(BaseModel):
    session_id: str
    question: str
    use_bm25: bool = False
    use_cross_encoder: bool = True

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be empty")
        return v


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: list
    latency: dict


@router.post("", response_model=AskResponse)
def ask_question(req: AskRequest, db: Session = Depends(get_db)):
    try:
        result = rag_pipeline(
            session_id=req.session_id,
            question=req.question,
            use_bm25=req.use_bm25,
            use_cross_encoder=req.use_cross_encoder,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = result.get("latency", {})

    q_row = Query(
        session_id=req.session_id,
        raw_question=req.question,
        standalone_question=result["question"],
    )
    db.add(q_row)
    db.flush()

    db.add(Response(
        query_id=q_row.id,
        answer=result["answer"],
        confidence=result["confidence"],
        sources=result["sources"],
    ))
    db.add(Analytics(
        query_id=q_row.id,
        retrieval_latency_ms=latency.get("retrieval_ms", 0),
        rerank_latency_ms=latency.get("rerank_ms", 0),
        llm_latency_ms=latency.get("llm_ms", 0),
        total_latency_ms=latency.get("total_ms", 0),
        num_sources=len(result["sources"]),
        used_bm25=req.use_bm25,
        used_cross_encoder=req.use_cross_encoder,
    ))
    db.commit()

    return AskResponse(**result)


@router.delete("/session/{session_id}")
def clear_session(session_id: str):
    memory.clear(session_id)
    return {"cleared": session_id}
