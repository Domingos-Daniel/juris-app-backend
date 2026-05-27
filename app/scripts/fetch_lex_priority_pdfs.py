from __future__ import annotations

import json
from pathlib import Path

import requests

from app.core.logger import configure_logging
from app.services.catalog import lex_ao_catalog_service
from app.services.pdf.ingestion import OFFICIAL_DOCUMENT_CATALOG


CATALOG_BY_SLUG = {item["diploma_slug"]: item for item in OFFICIAL_DOCUMENT_CATALOG}


def _safe_download(url: str, destination: Path) -> bool:
    response = requests.get(url, timeout=90, allow_redirects=True)
    if response.status_code != 200:
        return False
    if len(response.content) < 1000:
        return False
    destination.write_bytes(response.content)
    return True


def main() -> None:
    configure_logging()
    payload_path = lex_ao_catalog_service.documents_cache
    if not payload_path.exists():
        raise SystemExit("Catálogo do lex.ao ainda não foi gerado. Execute build_lex_catalog primeiro.")

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    targets = lex_ao_catalog_service.priority_targets(payload)
    out_dir = Path("C:/Projectos/TCC/legislacao")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for item in targets:
        slug = item.get("matched_internal_slug")
        pdf_url = item.get("download_pdf_url")
        if not slug or not pdf_url or slug not in CATALOG_BY_SLUG:
            results.append({"slug": slug, "status": "missing_pdf_url_or_catalog"})
            continue
        destination = out_dir / CATALOG_BY_SLUG[slug]["filename"]
        ok = _safe_download(pdf_url, destination)
        results.append({
            "slug": slug,
            "title": item.get("title"),
            "pdf_url": pdf_url,
            "destination": str(destination),
            "status": "downloaded" if ok else "failed",
        })

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
