from __future__ import annotations

import json
from pathlib import Path

from app.core.config import BASE_DIR
from app.core.logger import configure_logging
from app.db.postgres import postgres_manager

DATA_DIR = BASE_DIR / "data"


def _resolve_local_pdf_path(doc: dict) -> str | None:
    root = DATA_DIR / "raw_pdfs" / "lex_ao_bulk"
    entity = doc.get("entity_slug") or "unknown-entity"
    year = doc.get("year") or "unknown-year"
    slug = doc.get("document_slug") or "unknown-document"
    path = root / entity / year / f"{slug}.pdf"
    return str(path) if path.exists() else None


def main() -> None:
    configure_logging()
    catalog_path = DATA_DIR / "catalogs" / "lex_ao_documents.json"
    manifest_path = DATA_DIR / "catalogs" / "lex_ao_bulk_download_manifest.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")).get("documents", [])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    invalid_urls = {
        item["pdf_url"]
        for item in manifest.get("results", [])
        if item.get("status") == "too_small"
    }
    for item in catalog:
        if item.get("download_pdf_url") in invalid_urls:
            item["source_invalid"] = True
    imported = postgres_manager.import_legal_documents(
        catalog, local_path_resolver=_resolve_local_pdf_path
    )
    print(f"Imported {imported} lex.ao catalog records into Postgres.")


if __name__ == "__main__":
    main()
