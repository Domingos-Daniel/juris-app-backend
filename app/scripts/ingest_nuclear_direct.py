from __future__ import annotations

import asyncio
import gc
from pathlib import Path

from app.core.config import get_settings
from app.core.logger import configure_logging, get_logger
from app.services.pdf.ingestion import (
    OFFICIAL_DOCUMENT_CATALOG,
    legislation_ingestion_service,
    _document_spec,
)
from app.services.rag.vector_store import legislation_vector_store

logger = get_logger(__name__)


async def _main() -> None:
    configure_logging()
    settings = get_settings()

    existing_slugs = legislation_vector_store.available_diploma_slugs()
    logger.info("Diplomas ja ingeridos: %s", len(existing_slugs))

    total_chunks = 0
    processed = 0
    skipped = 0

    for i, item in enumerate(OFFICIAL_DOCUMENT_CATALOG, start=1):
        slug = item.get("diploma_slug", "")
        if slug and slug in existing_slugs:
            logger.info(
                "[%s/%s] Ignorando (ja existe): %s",
                i,
                len(OFFICIAL_DOCUMENT_CATALOG),
                item["title"],
            )
            skipped += 1
            continue

        pdf_path = settings.raw_pdfs_dir / item["filename"]
        if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
            logger.warning(
                "[%s/%s] Ficheiro nao encontrado: %s",
                i,
                len(OFFICIAL_DOCUMENT_CATALOG),
                pdf_path,
            )
            continue

        logger.info(
            "[%s/%s] Processando: %s (%s KB)",
            i,
            len(OFFICIAL_DOCUMENT_CATALOG),
            item["title"],
            pdf_path.stat().st_size // 1024,
        )
        try:
            result = await legislation_ingestion_service._ingest_document(
                {
                    "path": pdf_path,
                    "title": item["title"],
                    "url": item.get("url", ""),
                    "diploma_slug": slug,
                    "filename": item["filename"],
                }
            )
            processed += 1
            total_chunks += result.chunks_created
            logger.info(
                "[%s/%s] OK: %s chunks",
                i,
                len(OFFICIAL_DOCUMENT_CATALOG),
                result.chunks_created,
            )
        except Exception as exc:
            logger.error("[%s/%s] ERRO: %s", i, len(OFFICIAL_DOCUMENT_CATALOG), exc)

        gc.collect()

    logger.info(
        "Concluido: %s processados, %s ignorados, %s chunks total",
        processed,
        skipped,
        total_chunks,
    )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
