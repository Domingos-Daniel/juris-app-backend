from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytesseract
import requests
from pdf2image import convert_from_bytes

from app.core.config import get_settings


TERMS = (
    "impugnação",
    "impugnacao",
    "acto administrativo",
    "ato administrativo",
    "acto devido",
    "ato devido",
    "processo administrativo",
    "procedimento administrativo",
    "contencioso administrativo",
    "segunda via",
    "bilhete de identidade",
    "emolumento",
    "taxa",
    "identificação civil",
    "identificacao civil",
)


def ocr_first_page(pdf_bytes: bytes) -> str:
    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    images = convert_from_bytes(
        pdf_bytes,
        dpi=200,
        first_page=1,
        last_page=1,
        poppler_path=settings.poppler_bin_path,
    )
    if not images:
        return ""
    try:
        return pytesseract.image_to_string(images[0], lang=settings.ocr_language).strip()
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(images[0], lang="eng").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="assembleia-nacional")
    parser.add_argument("--years", nargs="*", default=["2022", "2023", "2024"])
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    payload = json.loads(Path("C:/Projectos/TCC/backend/data/catalogs/lex_ao_documents.json").read_text(encoding="utf-8"))
    rows = [
        item
        for item in payload.get("documents", [])
        if item.get("entity_slug") == args.entity and item.get("year") in args.years and str(item.get("document_slug", "")).startswith("lei-")
    ]
    checked = 0
    for row in rows:
        if checked >= args.limit:
            break
        pdf_url = row.get("download_pdf_url")
        if not pdf_url:
            continue
        checked += 1
        try:
            pdf_bytes = requests.get(pdf_url, timeout=120).content
            text = ocr_first_page(pdf_bytes)
        except Exception as exc:
            print(json.dumps({"slug": row.get("document_slug"), "title": row.get("title"), "error": str(exc)}, ensure_ascii=False))
            continue
        hay = re.sub(r"\s+", " ", text).lower()
        found = [term for term in TERMS if term in hay]
        if found:
            print(json.dumps({
                "slug": row.get("document_slug"),
                "title": row.get("title"),
                "pdf_url": pdf_url,
                "found": found,
                "excerpt": text[:1200],
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
