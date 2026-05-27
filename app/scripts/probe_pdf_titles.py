from __future__ import annotations

import argparse
from pathlib import Path

import fitz
import pytesseract
from pdf2image import convert_from_bytes

from app.core.config import get_settings


def ocr_first_pages(pdf_path: Path, max_pages: int = 2) -> str:
    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    pdf_bytes = pdf_path.read_bytes()
    images = convert_from_bytes(
        pdf_bytes,
        dpi=220,
        first_page=1,
        last_page=max_pages,
        poppler_path=settings.poppler_bin_path,
    )
    chunks: list[str] = []
    for image in images:
        try:
            text = pytesseract.image_to_string(image, lang=settings.ocr_language).strip()
        except pytesseract.TesseractError:
            text = pytesseract.image_to_string(image, lang="eng").strip()
        chunks.append(text)
    return "\n\n".join(chunk for chunk in chunks if chunk)


def native_first_pages(pdf_path: Path, max_pages: int = 2) -> str:
    document = fitz.open(pdf_path)
    chunks: list[str] = []
    for index, page in enumerate(document, start=1):
        if index > max_pages:
            break
        text = page.get_text("text").strip()
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--pages", type=int, default=2)
    args = parser.parse_args()

    for raw in args.paths:
        path = Path(raw)
        print(f"FILE {path}")
        try:
            native = native_first_pages(path, max_pages=args.pages)
            print("NATIVE:")
            print(native[:4000])
        except Exception as exc:
            print(f"NATIVE_ERROR: {exc}")
        print("---")
        try:
            ocr = ocr_first_pages(path, max_pages=args.pages)
            print("OCR:")
            print(ocr[:4000])
        except Exception as exc:
            print(f"OCR_ERROR: {exc}")
        print("=====")


if __name__ == "__main__":
    main()
