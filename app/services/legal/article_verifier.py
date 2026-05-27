from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.rag.vector_store import legislation_vector_store

_ARTICLE_DIGIT_RE = re.compile(
    r"(?:art|artigo|artigos|art\.?º?)\s*(\d+[.]?\d*)", re.IGNORECASE
)


@dataclass(slots=True)
class VerifiedArticle:
    article: str
    diploma: str
    diploma_slug: str
    status: str
    page: int | None = None
    chunk_text: str | None = None


class ArticleVerifier:
    async def verify(
        self, diploma_slug: str, article_number: str, expected_page: int | None = None
    ) -> VerifiedArticle | None:
        normalized_target = str(article_number).replace(".", "")
        chunks = legislation_vector_store.find_article_chunks(
            diploma_slug,
            article_number,
            expected_page=expected_page,
            limit=10,
        )

        if not chunks:
            where: dict[str, str] = {"diploma_slug": diploma_slug}
            search_query = f"artigo {article_number} {diploma_slug.replace('-', ' ')}"
            chunks = await legislation_vector_store.query(
                search_query,
                k=16,
                where=where,
            )

        for chunk in chunks:
            meta = chunk.metadata or {}

            refs = set(
                str(r).replace(".", "") for r in (meta.get("article_references") or [])
            )
            article_main = str(meta.get("article_main", "")).replace(".", "")
            article_number_field = str(meta.get("article_number", "")).replace(".", "")

            if (
                normalized_target in refs
                or normalized_target == article_main
                or normalized_target in article_number_field.split(",")
            ):
                chunk_page = chunk.page
                status = (
                    "confirmed"
                    if (expected_page is None or chunk_page == expected_page)
                    else "mismatched_page"
                )
                return VerifiedArticle(
                    article=article_number,
                    diploma=chunk.title or diploma_slug,
                    diploma_slug=diploma_slug,
                    status=status,
                    page=chunk_page,
                    chunk_text=chunk.text[:500],
                )

            text_refs = {
                m.group(1).replace(".", "")
                for m in _ARTICLE_DIGIT_RE.finditer(chunk.text or "")
            }
            if normalized_target in text_refs:
                chunk_page = chunk.page
                status = (
                    "confirmed_in_text"
                    if (expected_page is None or chunk_page == expected_page)
                    else "mismatched_page"
                )
                return VerifiedArticle(
                    article=article_number,
                    diploma=chunk.title or diploma_slug,
                    diploma_slug=diploma_slug,
                    status="confirmed_in_text",
                    page=chunk_page,
                    chunk_text=chunk.text[:500],
                )

        return None

    async def verify_batch(
        self,
        articles: list[tuple[str, str, int | None]],
    ) -> list[VerifiedArticle]:
        results: list[VerifiedArticle] = []
        for article, diploma_slug, page in articles:
            result = await self.verify(diploma_slug, article, page)
            if result is None:
                results.append(
                    VerifiedArticle(
                        article=article,
                        diploma=diploma_slug,
                        diploma_slug=diploma_slug,
                        status="not_found",
                    )
                )
            else:
                results.append(result)
        return results


article_verifier = ArticleVerifier()
