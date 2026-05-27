from __future__ import annotations

from app.core.config import get_settings
from app.db.models import RetrievedChunk
from app.services.rag.vector_store import legislation_vector_store


class RetrieverService:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def retrieve(self, query: str, k: int | None = None, where: dict | None = None) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []
        base_k = k or self.settings.retrieval_k
        candidate_k = max(base_k * 2, 8)
        return await legislation_vector_store.query(query=query, k=candidate_k, where=where)


retriever_service = RetrieverService()
