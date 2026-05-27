"""Migrate embeddings from MiniLM (384-d) to multilingual-e5-large (768-d).

Does NOT re-ingest PDFs — reads existing text_content from legal_segments
and recomputes only the embedding column.
"""

from __future__ import annotations

import asyncio
import gc
import time

from app.core.logger import configure_logging, get_logger
from app.db.postgres import postgres_manager
from app.services.rag.embeddings import embedding_service

logger = get_logger(__name__)

BATCH_SIZE = 64
COMMIT_EVERY = 1
GC_EVERY = 500


async def _main() -> None:
    configure_logging()
    postgres_manager.initialize()

    logger.info(
        "A carregar modelo multilingual-e5-large (1a vez demora ~2 min para download)..."
    )
    embedding_service.initialize()
    logger.info(
        "Modelo carregado. Dimensoes: %s",
        len(await embedding_service.embed_query("teste")),
    )

    # Count rows
    with postgres_manager.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM legal_segments WHERE text_content IS NOT NULL AND text_content != ''"
        )
        total = cur.fetchone()["cnt"]
    logger.info("Total de segmentos com texto: %s", total)

    if total == 0:
        logger.info("Nada a migrar.")
        return

    # Also count already-migrated (768-d embeddings)
    with postgres_manager.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM legal_segments WHERE array_length(embedding, 1) = 768"
        )
        already = cur.fetchone()["cnt"]
    if already > 0:
        logger.info("Ja existem %s segmentos com 768-d. Ignorando estes.", already)

    processed = 0
    start_time = time.monotonic()

    while True:
        with postgres_manager.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, text_content FROM legal_segments "
                "WHERE text_content IS NOT NULL AND text_content != '' "
                "AND (embedding IS NULL OR array_length(embedding, 1) != 768) "
                "ORDER BY id LIMIT %s",
                (BATCH_SIZE,),
            )
            rows = cur.fetchall()

        if not rows:
            break

        texts = [row["text_content"] for row in rows]
        ids = [row["id"] for row in rows]

        embeddings = await embedding_service.embed_texts(texts)

        with postgres_manager.connection() as conn, conn.cursor() as cur:
            for row_id, emb in zip(ids, embeddings):
                cur.execute(
                    "UPDATE legal_segments SET embedding = %s::double precision[] WHERE id = %s",
                    (list(emb), row_id),
                )
            conn.commit()

        processed += len(rows)

        elapsed = time.monotonic() - start_time
        pct = min(100, processed / total * 100)
        rate = processed / max(elapsed, 1)
        eta = (total - processed) / max(rate, 0.01)
        logger.info(
            "Migrados %s/%s (%.1f%%) | %.1f seg/s | ETA %.1f min",
            processed,
            total,
            pct,
            rate,
            eta / 60,
        )

        if processed % GC_EVERY < BATCH_SIZE:
            gc.collect()

    elapsed = time.monotonic() - start_time
    logger.info("Migracao concluida: %s segmentos em %.1f min", processed, elapsed / 60)

    # Verify
    with postgres_manager.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM legal_segments WHERE array_length(embedding, 1) = 768"
        )
        e5_count = cur.fetchone()["cnt"]
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM legal_segments WHERE array_length(embedding, 1) = 384"
        )
        old_count = cur.fetchone()["cnt"]
    logger.info("Embeddings 768-d: %s | 384-d restantes: %s", e5_count, old_count)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
