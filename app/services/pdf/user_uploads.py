from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings
from app.core.logger import get_logger
from app.db.models import UserDocumentItem
from app.db.postgres import postgres_manager
from app.services.pdf.chunker import semantic_chunk_text
from app.services.pdf.extractor import extract_pages_from_pdf
from app.services.pdf.ingestion import _primary_article_number, _is_normative_chunk, _normative_density
from app.services.rag.vector_store import legislation_vector_store

logger = get_logger(__name__)

MAX_UPLOAD_BYTES = 1 * 1024 * 1024


def _user_document_branch(filename: str, text: str) -> str:
    haystack = f"{filename} {text}".lower()
    if any(token in haystack for token in ["trabalho", "trabalhador", "empregador", "despedimento"]):
        return "laboral"
    if any(token in haystack for token in ["crime", "penal", "arguido", "queixa"]):
        return "penal"
    if any(token in haystack for token in ["mútuo", "mutuo", "empréstimo", "emprestimo", "contrato"]):
        return "civil"
    if any(token in haystack for token in ["constituição", "constituicao"]):
        return "constitucional"
    return "indeterminado"


def _user_document_metadata(
    display_name: str, page_number: int, chunk: str, page_used_ocr: bool, chunk_index: int
) -> dict:
    return {
        "source": display_name,
        "title": display_name,
        "link_original": None,
        "page": page_number,
        "article_number": _primary_article_number(chunk),
        "law_status": "Documento do utilizador",
        "used_ocr": page_used_ocr,
        "chunk_index": chunk_index,
        "source_scope": "user_upload",
        "legal_branch": _user_document_branch(display_name, chunk),
        "diploma_slug": None,
        "is_front_matter": False,
        "is_structural": False,
        "normative_density": _normative_density(chunk),
        "is_normative": _is_normative_chunk(chunk),
        "source_priority": 0.4,
        "document_kind": "user_document",
    }


def _guess_category(filename: str, text: str) -> str:
    haystack = f"{filename} {text}".lower()
    if any(token in haystack for token in ["contrato", "acordo", "aceitacao", "aceitação"]):
        return "Contrato"
    if any(token in haystack for token in ["trabalho", "empregador", "trabalhador"]):
        return "Trabalho"
    if any(token in haystack for token in ["procura", "mandato", "procuracao", "procuração"]):
        return "Mandato"
    if any(token in haystack for token in ["peticao", "petição", "requerimento"]):
        return "Peca Processual"
    return "Geral"


def _quality_status(used_ocr: bool, chunk_count: int, pages_count: int) -> str:
    if chunk_count <= 0:
        return "empty"
    if used_ocr:
        return "ocr"
    if pages_count > 0 and chunk_count <= max(1, pages_count // 3):
        return "partial"
    return "good"


class UserDocumentService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.upload_dir = self.settings.processed_dir / "user_uploads_tmp"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir = self.settings.processed_dir / "user_documents"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def save_uploaded_pdf(self, upload: UploadFile, user_id: str | int | None = None) -> UserDocumentItem:
        if upload.content_type != "application/pdf":
            raise ValueError("Apenas ficheiros PDF sao suportados neste momento.")
        payload = await upload.read()
        if not payload:
            raise ValueError("O ficheiro PDF esta vazio.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise ValueError("O PDF excede o limite maximo de 1 MB.")
        safe_name = Path(upload.filename or "documento.pdf").name
        file_hash = hashlib.sha1((safe_name + str(len(payload))).encode("utf-8")).hexdigest()
        
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        tmp_path = self.upload_dir / f"{file_hash}.pdf"
        tmp_path.write_bytes(payload)
        persistent_path = self.storage_dir / f"{file_hash}-{safe_name}"
        persistent_path.write_bytes(payload)
        try:
            return await self._process_pdf_path(
                tmp_path,
                safe_name,
                len(payload),
                user_id=user_id,
                storage_path=str(persistent_path),
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    async def reprocess_document(self, document_id: str, user_id: str | int | None = None) -> UserDocumentItem:
        document = postgres_manager.get_document(document_id, user_id=user_id)
        if not document:
            raise ValueError("Documento nao encontrado.")
        storage_path = document.get("storage_path")
        if not storage_path:
            raise ValueError("Este documento foi criado antes da preservacao do PDF original e nao pode ser reprocessado automaticamente.")
        pdf_path = Path(storage_path)
        if not pdf_path.exists():
            raise ValueError("O PDF original deste documento ja nao esta disponivel no armazenamento local.")
        legislation_vector_store.delete_document_chunks(document_id)
        postgres_manager.delete_document(document_id, user_id=user_id)
        return await self._process_pdf_path(
            pdf_path,
            document.get("display_name") or document.get("filename") or pdf_path.name,
            int(document.get("size_bytes") or pdf_path.stat().st_size),
            user_id=user_id,
            storage_path=str(pdf_path),
        )

    async def _process_pdf_path(
        self,
        pdf_path: Path,
        display_name: str,
        size_bytes: int,
        user_id: str | int | None = None,
        storage_path: str | None = None,
    ) -> UserDocumentItem:
        pages, used_ocr = extract_pages_from_pdf(pdf_path)
        chunks_payload: list[dict] = []
        ocr_pages = 0
        joined_parts: list[str] = []
        for page_info in pages:
            page_number = page_info["page"]
            page_text = page_info["text"].strip()
            page_used_ocr = bool(page_info.get("used_ocr", False))
            if page_used_ocr:
                ocr_pages += 1
            if not page_text:
                continue
            joined_parts.append(page_text)
            for chunk_index, chunk in enumerate(semantic_chunk_text(page_text), start=1):
                chunk_id = hashlib.sha1(
                    f"user:{display_name}:{page_number}:{chunk_index}:{chunk[:120]}".encode("utf-8")
                ).hexdigest()
                chunks_payload.append(
                    {
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": _user_document_metadata(
                            display_name=display_name,
                            page_number=page_number,
                            chunk=chunk,
                            page_used_ocr=page_used_ocr,
                            chunk_index=chunk_index,
                        ),
                    }
                )
        if not chunks_payload:
            raise ValueError("Nao foi possivel extrair texto util deste PDF.")
        full_text = "\n\n".join(joined_parts).strip()
        summary = full_text[:320] if full_text else None
        preview_text = full_text[:2000] if full_text else None
        category = _guess_category(display_name, full_text[:1200])
        quality_status = _quality_status(used_ocr, len(chunks_payload), len(pages))
        document_id = postgres_manager.save_document(
            filename=display_name,
            display_name=display_name,
            storage_path=storage_path,
            mime_type="application/pdf",
            size_bytes=size_bytes,
            status="processed",
            page_count=len(pages),
            chunks_created=len(chunks_payload),
            extraction_mode="ocr" if used_ocr else "direct",
            quality_status=quality_status,
            summary=summary,
            preview_text=preview_text,
            category=category,
            user_id=user_id,
        )
        for item in chunks_payload:
            item["metadata"]["document_id"] = document_id
        await legislation_vector_store.upsert_documents(chunks_payload)
        logger.info(
            "Documento do utilizador %s processado, %s chunks criados, %s paginas via OCR",
            display_name,
            len(chunks_payload),
            ocr_pages,
        )
        payload = postgres_manager.get_document(document_id, user_id=user_id)
        if not payload:
            raise ValueError("Documento processado mas nao foi possivel reler o registo persistido.")
        return UserDocumentItem(**payload)

    def list_documents(self, user_id: str | int | None = None) -> list[UserDocumentItem]:
        return [UserDocumentItem(**row) for row in postgres_manager.list_documents(user_id=user_id)]

    def get_document(self, document_id: str, user_id: str | int | None = None) -> UserDocumentItem:
        payload = postgres_manager.get_document(document_id, user_id=user_id)
        if not payload:
            raise ValueError("Documento nao encontrado.")
        return UserDocumentItem(**payload)

    def rename_document(self, document_id: str, display_name: str, user_id: str | int | None = None) -> UserDocumentItem:
        postgres_manager.rename_document(document_id, display_name, user_id=user_id)
        payload = postgres_manager.get_document(document_id, user_id=user_id)
        if not payload:
            raise ValueError("Documento nao encontrado apos renomeacao.")
        return UserDocumentItem(**payload)

    def delete_document(self, document_id: str, user_id: str | int | None = None) -> None:
        payload = postgres_manager.get_document(document_id, user_id=user_id)
        legislation_vector_store.delete_document_chunks(document_id)
        postgres_manager.delete_document(document_id, user_id=user_id)
        storage_path = (payload or {}).get("storage_path")
        if storage_path:
            try:
                Path(storage_path).unlink(missing_ok=True)
            except Exception:
                pass

    def get_document_preview(self, document_id: str, user_id: str | int | None = None) -> dict:
        payload = postgres_manager.get_document(document_id, user_id=user_id)
        if not payload:
            raise ValueError("Documento nao encontrado.")
        chunks = legislation_vector_store.get_document_chunks(document_id, limit=6)
        return {
            "document": payload,
            "chunks": [
                {
                    "page": chunk.page,
                    "text": chunk.text,
                    "article_number": chunk.article_number,
                }
                for chunk in chunks
            ],
        }

    def attach_document_to_chat(
        self, document_id: str, chat_id: str | None, user_id: str | int | None = None
    ) -> dict:
        if chat_id:
            postgres_manager.set_chat_active_document(chat_id, document_id, user_id=user_id)
        postgres_manager.mark_document_used(document_id, user_id=user_id)
        payload = postgres_manager.get_document(document_id, user_id=user_id)
        if not payload:
            raise ValueError("Documento nao encontrado.")
        return payload


user_document_service = UserDocumentService()
