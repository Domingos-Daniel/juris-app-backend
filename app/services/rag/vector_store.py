from __future__ import annotations

from app.db.models import RetrievedChunk
from app.db.postgres import postgres_manager
from app.services.rag.embeddings import embedding_service


class LegislationVectorStore:
    async def reset_collection(self) -> None:
        postgres_manager.initialize()
        postgres_manager.delete_segments_by_metadata()

    async def upsert_documents(self, items: list[dict]) -> int:
        if not items:
            return 0
        texts = [item["text"] for item in items]
        embeddings = await embedding_service.embed_texts(texts)
        enriched = []
        for item, vector in zip(items, embeddings):
            enriched.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "embedding": vector,
                }
            )
        return postgres_manager.upsert_legal_segments(enriched)

    async def query(
        self, query: str, k: int, where: dict | None = None
    ) -> list[RetrievedChunk]:
        return await postgres_manager.query_legal_segments(query, k=k, where=where)

    def get_branch_prototypes(self):
        return postgres_manager.get_branch_prototypes()

    def get_document_chunks(self, document_id: str, limit: int = 8) -> list[RetrievedChunk]:
        return postgres_manager.get_document_chunks(document_id, limit=limit)

    def delete_document_chunks(self, document_id: str) -> None:
        postgres_manager.delete_document_chunks(document_id)

    def delete_by_metadata(self, **metadata_filters) -> int:
        return postgres_manager.delete_segments_by_metadata(**metadata_filters)

    def count_by_metadata(self, **metadata_filters) -> int:
        return postgres_manager.count_segments_by_metadata(**metadata_filters)

    def available_diploma_slugs(self) -> set[str]:
        return postgres_manager.available_diploma_slugs()

    def find_article_chunks(
        self,
        diploma_slug: str,
        article_number: str,
        expected_page: int | None = None,
        limit: int = 8,
    ) -> list[RetrievedChunk]:
        return postgres_manager.find_article_chunks(
            diploma_slug=diploma_slug,
            article_number=article_number,
            expected_page=expected_page,
            limit=limit,
        )

    def get_stats(self) -> dict:
        return postgres_manager.get_legal_segment_stats()


legislation_vector_store = LegislationVectorStore()
