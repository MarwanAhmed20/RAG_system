from fastapi import APIRouter
from core.loader import health_status
from core.rag import cache_stats

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
def health_check():
    status = health_status()
    status["cache"] = cache_stats()
    return status