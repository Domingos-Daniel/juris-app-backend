from __future__ import annotations

from pathlib import Path

import fitz

from app.core.logger import get_logger
from app.services.pdf.ocr import ocr_service


logger = get_logger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, bool]:
    document = fitz.open(pdf_path)
    pages: list[str] = []
    for page in document:
        pages.append(page.get_text("text").strip())

    joined = "\n\n".join(page for page in pages if page)
    if joined.strip():
        document.close()
        return joined, False

    try:
        results = ocr_service.ocr_pdf_pages(pdf_path)
        ocr_text = "\n".join(page["text"] for page in results if page["text"])
        document.close()
        return ocr_text, True
    except Exception as exc:
        logger.warning("OCR indisponivel para %s: %s", pdf_path.name, exc)
        document.close()
        return "", False


def extract_pages_from_pdf(pdf_path: Path) -> tuple[list[dict], bool]:
    document = fitz.open(pdf_path)
    pages: list[dict] = []
    needs_ocr = False
    for index, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        used_ocr = len(text) < 80
        if used_ocr:
            needs_ocr = True
        pages.append({"page": index, "text": text, "used_ocr": False})

    if not needs_ocr:
        document.close()
        return pages, False

    try:
        pages_to_ocr = [p["page"] for p in pages if len(p["text"]) < 80]
        ocr_results = ocr_service.ocr_pdf_pages(pdf_path, pages_to_ocr=pages_to_ocr)
        ocr_pages = {item["page"]: item for item in ocr_results}
        merged_pages: list[dict] = []
        for page in pages:
            if len(page["text"]) >= 80:
                merged_pages.append(page)
                continue
            ocr_page = ocr_pages.get(page["page"])
            if ocr_page and ocr_page.get("text", "").strip():
                merged_pages.append(
                    {
                        "page": page["page"],
                        "text": ocr_page["text"].strip(),
                        "used_ocr": True,
                    }
                )
            else:
                merged_pages.append(page)
        document.close()
        return merged_pages, True
    except Exception as exc:
        logger.warning("OCR indisponivel para %s: %s", pdf_path.name, exc)
        document.close()
        return pages, False


def extract_pages_from_pdf_range(pdf_path: Path, first_page: int, last_page: int) -> tuple[list[dict], bool]:
    document = fitz.open(pdf_path)
    pages: list[dict] = []
    needs_ocr = False
    for index in range(first_page, min(last_page, len(document)) + 1):
        page = document[index - 1]
        text = page.get_text("text").strip()
        used_ocr = len(text) < 80
        if used_ocr:
            needs_ocr = True
        pages.append({"page": index, "text": text, "used_ocr": False})

    if not needs_ocr:
        document.close()
        return pages, False

    try:
        pages_to_ocr = [p["page"] for p in pages if len(p["text"]) < 80]
        ocr_results = ocr_service.ocr_pdf_pages(pdf_path, pages_to_ocr=pages_to_ocr)
        ocr_pages = {item["page"]: item for item in ocr_results}
        merged_pages: list[dict] = []
        for page in pages:
            if len(page["text"]) >= 80:
                merged_pages.append(page)
                continue
            ocr_page = ocr_pages.get(page["page"])
            if ocr_page and ocr_page.get("text", "").strip():
                merged_pages.append(
                    {
                        "page": page["page"],
                        "text": ocr_page["text"].strip(),
                        "used_ocr": True,
                    }
                )
            else:
                merged_pages.append(page)
        document.close()
        return merged_pages, True
    except Exception as exc:
        logger.warning("OCR por intervalo indisponivel para %s: %s", pdf_path.name, exc)
        document.close()
        return pages, False
