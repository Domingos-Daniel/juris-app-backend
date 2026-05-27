from __future__ import annotations

import logging
import os
import concurrent.futures
import gc
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.logger import get_logger

# Silenciar logs excessivos do Paddle
logging.getLogger("ppocr").setLevel(logging.WARNING)

logger = get_logger(__name__)


class OCRService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._engine = None

    def reset_engine(self) -> None:
        """
        Força a liberação de memória do PaddleOCR deletando a instância do motor.
        """
        if self._engine is not None:
            logger.info("Reiniciando motor OCR para liberar memoria...")
            del self._engine
            self._engine = None
            gc.collect()

    @property
    def engine(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR

                self._engine = PaddleOCR(
                    use_angle_cls=True, lang=self.settings.ocr_language
                )
                logger.info("Motor OCR: paddleocr")
            except Exception as exc:
                logger.info("PaddleOCR indisponivel (%s), a usar tesseract", exc)
                self._engine = "tesseract"
        return self._engine

    def _ocr_paddle(self, img_np: Any) -> list[str]:
        result = self.engine.ocr(img_np, cls=True)
        if result and result[0]:
            return [line[1][0] for line in result[0]]
        return []

    def _ocr_tesseract(self, img: Image.Image) -> str:
        import pytesseract

        if self.settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.settings.tesseract_cmd
        if self.settings.tesseract_data_dir:
            import os

            os.environ["TESSDATA_PREFIX"] = self.settings.tesseract_data_dir
        lang = self.settings.ocr_language or "por"
        text = pytesseract.image_to_string(img, lang=lang)
        return text

    def ocr_pdf_pages(
        self, pdf_path: Path, pages_to_ocr: list[int] | None = None
    ) -> list[dict]:
        """
        Executa OCR em páginas específicas de um PDF usando PaddleOCR.
        Processa em lotes para evitar picos de memória e instabilidade.
        """
        doc = fitz.open(pdf_path)
        indices = (
            pages_to_ocr if pages_to_ocr is not None else list(range(1, len(doc) + 1))
        )

        results: list[dict] = []

        # Processar em lotes de 5 páginas para segurança de memória
        batch_size = 5
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]

            for page_num in batch_indices:
                try:
                    page = doc[page_num - 1]
                    pix = page.get_pixmap(dpi=self.settings.ocr_render_dpi)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    if self.engine == "tesseract":
                        text = self._ocr_tesseract(img).strip()
                        results.append({"page": page_num, "text": text})
                    else:
                        img_np = np.array(img)
                        lines = self._ocr_paddle(img_np)
                        results.append(
                            {"page": page_num, "text": "\n".join(lines).strip()}
                        )
                        del img_np

                    del pix
                    del img
                    gc.collect()
                except Exception as exc:
                    logger.error(
                        "Erro no OCR da página %s de %s: %s",
                        page_num,
                        pdf_path.name,
                        exc,
                    )
                    results.append({"page": page_num, "text": ""})

        doc.close()
        return results

    def ocr_pdf_pages_range(
        self, pdf_path: Path, first_page: int, last_page: int
    ) -> list[dict]:
        return self.ocr_pdf_pages(pdf_path, list(range(first_page, last_page + 1)))


ocr_service = OCRService()


# Mantendo compatibilidade de interface se necessário, mas o ideal é usar o ocr_service
def ocr_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    # Implementação temporária para compatibilidade se o arquivo de bytes for passado
    # O ideal é passar o Path para evitar salvar em disco desnecessariamente,
    # mas o fitz aceita stream de bytes.
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # Salvar temporário ou usar stream
    temp_path = Path("temp_ocr.pdf")
    temp_path.write_bytes(pdf_bytes)
    res = ocr_service.ocr_pdf_pages(temp_path)
    if temp_path.exists():
        temp_path.unlink()
    return res


def ocr_pdf_pages_range(
    pdf_bytes: bytes, first_page: int, last_page: int, dpi: int = 160
) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    temp_path = Path("temp_ocr_range.pdf")
    temp_path.write_bytes(pdf_bytes)
    res = ocr_service.ocr_pdf_pages_range(temp_path, first_page, last_page)
    if temp_path.exists():
        temp_path.unlink()
    return res
