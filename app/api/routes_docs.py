from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.auth import get_current_user
from app.db.models import (
    DocumentRenameRequest,
    DocumentUseRequest,
    IngestSummary,
    UserDocumentItem,
    UserDocumentListResponse,
)
from app.services.pdf.ingestion import legislation_ingestion_service
from app.services.pdf.user_uploads import user_document_service


router = APIRouter(tags=["documents"])


@router.post("/docs/ingest", response_model=IngestSummary)
async def ingest_documents(
    current_user: dict = Depends(get_current_user),
) -> IngestSummary:
    try:
        return await legislation_ingestion_service.ingest_official_documents()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Falha ao ingerir PDFs: {exc}"
        ) from exc


@router.get("/docs", response_model=UserDocumentListResponse)
async def list_user_documents(
    current_user: dict = Depends(get_current_user),
) -> UserDocumentListResponse:
    return UserDocumentListResponse(
        items=user_document_service.list_documents(user_id=current_user["id"])
    )


@router.get("/docs/{document_id}")
async def get_document(
    document_id: str, current_user: dict = Depends(get_current_user)
) -> UserDocumentItem:
    try:
        return user_document_service.get_document(
            document_id, user_id=current_user["id"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/docs/{document_id}/preview")
async def preview_document(
    document_id: str, current_user: dict = Depends(get_current_user)
) -> dict:
    try:
        return user_document_service.get_document_preview(
            document_id, user_id=current_user["id"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/docs/upload", response_model=UserDocumentItem)
async def upload_user_document(
    file: UploadFile = File(...), current_user: dict = Depends(get_current_user)
) -> UserDocumentItem:
    try:
        return await user_document_service.save_uploaded_pdf(
            file, user_id=current_user["id"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Falha ao processar PDF do utilizador: {exc}"
        ) from exc


@router.patch("/docs/{document_id}/rename", response_model=UserDocumentItem)
async def rename_document(
    document_id: str,
    payload: DocumentRenameRequest,
    current_user: dict = Depends(get_current_user),
) -> UserDocumentItem:
    try:
        return user_document_service.rename_document(
            document_id, payload.display_name, user_id=current_user["id"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/docs/{document_id}/use")
async def use_document(
    document_id: str,
    payload: DocumentUseRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    try:
        return user_document_service.attach_document_to_chat(
            document_id, payload.chat_id, user_id=current_user["id"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/docs/{document_id}/reprocess", response_model=UserDocumentItem)
async def reprocess_document(
    document_id: str, current_user: dict = Depends(get_current_user)
) -> UserDocumentItem:
    try:
        return await user_document_service.reprocess_document(
            document_id, user_id=current_user["id"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/docs/{document_id}")
async def delete_document(
    document_id: str, current_user: dict = Depends(get_current_user)
) -> dict:
    try:
        user_document_service.delete_document(document_id, user_id=current_user["id"])
        return {"ok": True, "document_id": document_id}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
