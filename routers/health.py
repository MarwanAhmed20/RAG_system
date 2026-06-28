from fastapi import APIRouter
from loader import health_status

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
def health_check():
    return health_status()
