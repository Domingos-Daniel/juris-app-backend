from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import get_current_user
from app.db.models import ChatListItem, ChatListResponse
from app.db.postgres import postgres_manager


router = APIRouter(tags=["chats"])


@router.get("/chats", response_model=ChatListResponse)
async def list_chats(
    current_user: dict = Depends(get_current_user),
) -> ChatListResponse:
    try:
        items = [
            ChatListItem(**item)
            for item in postgres_manager.list_chats(user_id=current_user["id"])
        ]
        return ChatListResponse(items=items)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Falha ao listar chats: {exc}"
        ) from exc


@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    try:
        deleted = postgres_manager.delete_chat(chat_id, user_id=current_user["id"])
        if not deleted:
            raise HTTPException(status_code=404, detail="Chat nao encontrado")
        return {"ok": True, "deleted": chat_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Falha ao eliminar chat: {exc}"
        ) from exc


@router.delete("/chats")
async def delete_all_chats(current_user: dict = Depends(get_current_user)):
    try:
        count = postgres_manager.delete_all_chats(user_id=current_user["id"])
        return {"ok": True, "deleted_count": count}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Falha ao eliminar chats: {exc}"
        ) from exc
