from __future__ import annotations

from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_auth import router as auth_router
from app.api.routes_chats import router as chats_router
from app.api.routes_chat import router as chat_router
from app.api.routes_docs import router as docs_router
from app.api.routes_catalog import router as catalog_router
from app.core.config import get_settings
from app.core.logger import configure_logging
from app.db.postgres import postgres_manager
from app.services.rag.embeddings import embedding_service
from app.services.llm.deepseek_client import deepseek_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    postgres_manager.initialize()
    embedding_service.initialize()
    # Shared httpx client with connection pooling — avoids creating a new TCP
    # connection per streaming request (major source of latency).
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    )
    deepseek_client.set_http_client(http_client)
    yield
    await http_client.aclose()


configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Backend RAG para assistencia juridica angolana baseada em PDFs legais.",
    docs_url="/api-docs",
    redoc_url="/api-redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(chats_router, prefix=settings.api_prefix)
app.include_router(docs_router, prefix=settings.api_prefix)
app.include_router(catalog_router, prefix=settings.api_prefix)
