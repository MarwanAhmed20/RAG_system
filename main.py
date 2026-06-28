from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.loader import load_all
from core.database import init_db
from routers import qa, health, evaluate


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_all()
    yield
    print("[main] Shutdown.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa.router)
app.include_router(health.router)
app.include_router(evaluate.router)
