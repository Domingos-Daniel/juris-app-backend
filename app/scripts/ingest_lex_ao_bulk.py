from __future__ import annotations

import asyncio
import hashlib
import json
import re
import gc
import time
from pathlib import Path

from app.core.config import BASE_DIR, get_settings
from app.core.logger import configure_logging, get_logger
from app.db.postgres import postgres_manager
from app.services.pdf.extractor import extract_pages_from_pdf
from app.services.pdf.chunker import legal_semantic_chunks

logger = get_logger(__name__)
ARTICLE_RE = re.compile(r"(?:art|artigo|artigos)\.?\s*(\d+)", re.IGNORECASE)

DATA_DIR = BASE_DIR / "data"
BULK_DIR = DATA_DIR / "raw_pdfs" / "lex_ao_bulk"
CATALOG_PATH = DATA_DIR / "catalogs" / "lex_ao_documents.json"
CHUNK_BATCH_SIZE = 20
GC_EVERY_N_PDFS = 10


def _extract_article_numbers(text: str) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for match in ARTICLE_RE.finditer(text or ""):
        num = match.group(1)
        if num not in seen:
            seen.add(num)
            items.append(num)
    return items


def _primary_article_number(text: str) -> str | None:
    refs = _extract_article_numbers(text)
    return refs[0] if refs else None


def _chunk_id(pdf_stem: str, page: int, chunk_idx: int, text_sample: str) -> str:
    return hashlib.sha1(
        f"{pdf_stem}:{page}:{chunk_idx}:{text_sample[:100]}".encode()
    ).hexdigest()


def _load_catalog_index() -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    if not CATALOG_PATH.exists():
        return index
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    for doc in data.get("documents", []):
        key = (doc.get("entity_slug") or "", doc.get("document_slug") or "")
        index[key] = doc
    return index


def _catalog_entry(
    entity_slug: str, document_slug: str, catalog_index: dict
) -> dict | None:
    return catalog_index.get((entity_slug, document_slug))


def _build_metadata(
    pdf_path: Path,
    entity_slug: str,
    year: str,
    document_slug: str,
    catalog: dict | None,
):
    stem = pdf_path.stem
    title = (
        catalog.get("document_title") or catalog.get("title") or stem
        if catalog
        else stem
    )
    return {
        "source": pdf_path.name,
        "title": title,
        "link_original": catalog.get("download_pdf_url") if catalog else None,
        "page": 0,
        "article_number": None,
        "law_status": catalog.get("law_status") or "Nao verificado"
        if catalog
        else "Nao verificado",
        "used_ocr": False,
        "chunk_index": 0,
        "source_scope": "official",
        "document_id": catalog.get("document_id") if catalog else None,
        "legal_branch": catalog.get("legal_branch_guess") if catalog else None,
        "diploma_slug": catalog.get("matched_internal_slug") or document_slug
        if catalog
        else document_slug,
        "is_front_matter": False,
        "is_structural": False,
        "normative_density": 0.0,
        "is_normative": False,
        "source_priority": 0.8,
        "document_kind": "legislation_official",
        "catalog_version": "lex-ao-bulk-v1",
        "document_role": "lei_especial",
        "norm_type": "misto",
        "topic_route": "geral",
        "primary_topics": [],
        "exclusive_domains": [],
        "strict_topics": [],
        "routing_priority": 0.8,
        "is_primary_source": False,
        "article_main": None,
        "article_references": [],
        "segmentation": "semantic_fallback",
        "page_is_context_heavy": False,
        "chunk_priority": 1.0,
        "chunk_kind": "text_normative",
    }


async def _ingest_bulk_document(
    pdf_path: Path,
    entity_slug: str,
    year: str,
    document_slug: str,
    catalog_index: dict,
) -> int:
    catalog = _catalog_entry(entity_slug, document_slug, catalog_index)
    try:
        pages, used_ocr = extract_pages_from_pdf(pdf_path)
    except Exception as exc:
        logger.warning("Falha ao extrair texto de %s: %s", pdf_path.name, exc)
        return 0

    chunks_payload: list[dict] = []
    base_meta = _build_metadata(pdf_path, entity_slug, year, document_slug, catalog)
    text_content = ""

    for page_info in pages:
        page_number = page_info["page"]
        page_text = (page_info.get("text") or "").strip()
        if not page_text:
            continue
        page_used_ocr = bool(page_info.get("used_ocr", False))
        text_content += page_text + "\n\n"

        for chunk_idx, chunk in enumerate(legal_semantic_chunks(page_text), start=1):
            chunk_text = chunk["text"].strip()
            if len(chunk_text) < 40:
                continue
            article_main = chunk.get("article_main") or _primary_article_number(
                chunk_text
            )
            article_refs = _extract_article_numbers(chunk_text)
            meta = dict(base_meta)
            meta["page"] = page_number
            meta["chunk_index"] = chunk_idx
            meta["article_number"] = (
                ", ".join(article_refs[:4]) if article_refs else None
            )
            meta["article_main"] = article_main
            meta["article_references"] = article_refs
            meta["used_ocr"] = page_used_ocr
            meta["segmentation"] = chunk.get("segmentation", "semantic_fallback")

            cid = _chunk_id(pdf_path.stem, page_number, chunk_idx, chunk_text)
            chunks_payload.append(
                {
                    "id": cid,
                    "text": chunk_text,
                    "metadata": meta,
                }
            )

    if not chunks_payload:
        logger.info("Sem chunks aproveitaveis em %s", pdf_path.name)
        return 0

    total = 0
    for batch_start in range(0, len(chunks_payload), CHUNK_BATCH_SIZE):
        batch = chunks_payload[batch_start : batch_start + CHUNK_BATCH_SIZE]
        texts = [item["text"] for item in batch]
        from app.services.rag.embeddings import embedding_service

        embeddings = await embedding_service.embed_texts(texts)
        enriched = []
        for item, emb in zip(batch, embeddings):
            enriched.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "embedding": emb,
                }
            )
        postgres_manager.upsert_legal_segments(enriched)
        total += len(enriched)

    catalog_title = catalog.get("document_title", "") if catalog else ""
    logger.info("Ingerido %s: %s chunks (%s)", pdf_path.name, total, catalog_title)
    return total


def _collect_pdfs() -> list[tuple[Path, str, str, str]]:
    pdfs: list[tuple[Path, str, str, str]] = []
    if not BULK_DIR.exists():
        return pdfs
    for ent_dir in sorted(BULK_DIR.iterdir()):
        if not ent_dir.is_dir():
            continue
        entity_slug = ent_dir.name
        for yr_dir in sorted(ent_dir.iterdir()):
            if not yr_dir.is_dir():
                continue
            year = yr_dir.name
            for pdf_file in sorted(yr_dir.glob("*.pdf")):
                doc_slug = pdf_file.stem
                pdfs.append((pdf_file, entity_slug, year, doc_slug))
    return pdfs


async def _main() -> None:
    configure_logging()
    settings = get_settings()
    postgres_manager.initialize()

    logger.info("A carregar indice do catalogo lex.ao...")
    catalog_index = _load_catalog_index()
    logger.info("Catalogo carregado: %s entradas", len(catalog_index))

    pdfs = _collect_pdfs()
    logger.info("PDFs encontrados em %s: %s", BULK_DIR, len(pdfs))

    ingested_slugs = postgres_manager.available_diploma_slugs()
    logger.info("Diplomas ja ingeridos: %s", len(ingested_slugs))

    total_chunks = 0
    total_pdfs = 0
    skipped = 0
    failed = 0
    start_time = time.monotonic()

    for i, (pdf_path, entity_slug, year, doc_slug) in enumerate(pdfs, start=1):
        if doc_slug and doc_slug in ingested_slugs:
            skipped += 1
            continue

        logger.info("[%s/%s] %s", i, len(pdfs), pdf_path.relative_to(BULK_DIR))
        try:
            chunks = await _ingest_bulk_document(
                pdf_path, entity_slug, year, doc_slug, catalog_index
            )
            total_chunks += chunks
            if chunks > 0:
                total_pdfs += 1
            else:
                failed += 1
        except Exception as exc:
            logger.error("Falha em %s: %s", pdf_path.name, exc)
            failed += 1

        if total_pdfs > 0 and total_pdfs % GC_EVERY_N_PDFS == 0:
            elapsed = time.monotonic() - start_time
            logger.info(
                "Progresso: %s PDFs, %s chunks, %.1f min",
                total_pdfs,
                total_chunks,
                elapsed / 60.0,
            )
            gc.collect()

    elapsed = time.monotonic() - start_time
    logger.info(
        "Ingestao concluida: %s PDFs processados, %s chunks, %s ignorados, %s falhas, %.1f min",
        total_pdfs,
        total_chunks,
        skipped,
        failed,
        elapsed / 60.0,
    )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
