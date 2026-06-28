from datetime import datetime
from sqlalchemy import Integer, String, Float, Text, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from core.database import Base


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    raw_question: Mapped[str] = mapped_column(Text)
    standalone_question: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    response: Mapped["Response"] = relationship("Response", back_populates="query", uselist=False)
    analytics: Mapped["Analytics"] = relationship("Analytics", back_populates="query", uselist=False)


class Response(Base):
    __tablename__ = "responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    query_id: Mapped[int] = mapped_column(ForeignKey("queries.id"), unique=True)
    answer: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    sources: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    query: Mapped["Query"] = relationship("Query", back_populates="response")


class Analytics(Base):
    __tablename__ = "analytics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    query_id: Mapped[int] = mapped_column(ForeignKey("queries.id"), unique=True)
    retrieval_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    rerank_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    llm_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    num_sources: Mapped[int] = mapped_column(Integer, default=0)
    used_bm25: Mapped[bool] = mapped_column(Boolean, default=False)
    used_cross_encoder: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    query: Mapped["Query"] = relationship("Query", back_populates="analytics")
