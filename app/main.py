"""
ResearchMind — FastAPI application factory.
"""

from contextlib import asynccontextmanager
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.middleware import (
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    RateLimitMiddleware,
)
from app.api.routes import ingest, query, agent

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic."""
    settings = get_settings()
    logger.info("researchmind_starting", env=settings.app_env)

    # Eagerly create the Qdrant collection so first ingest is fast
    from app.services.rag.vector_store import ensure_collection
    await ensure_collection(settings.qdrant_collection)
    logger.info("qdrant_collection_ready", collection=settings.qdrant_collection)

    yield

    logger.info("researchmind_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ResearchMind API",
        description="Agentic RAG pipeline with hybrid retrieval and ReAct agent",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware (order matters — first added = outermost wrapper) ──────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware, calls=60, period=60)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(agent.router)

    @app.get("/health", tags=["System"])
    async def health():
        return {"status": "ok", "env": settings.app_env}

    return app


app = create_app()
