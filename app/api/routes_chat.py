from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from app.core.auth import get_current_user
from app.db.models import ChatRequest, ChatResponse
from app.services.rag.pipeline import rag_pipeline

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest, current_user: dict = Depends(get_current_user)
) -> ChatResponse:
    try:
        return await rag_pipeline.answer_query(
            payload.question,
            provider=payload.provider,
            conversation_history=payload.conversation_history,
            chat_id=payload.chat_id,
            active_document_id=payload.active_document_id,
            user_id=current_user["id"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Falha ao responder pergunta: {exc}"
        ) from exc


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest, current_user: dict = Depends(get_current_user)
):
    try:
        return StreamingResponse(
            rag_pipeline.answer_query_stream_safe(
                payload.question,
                provider=payload.provider,
                conversation_history=payload.conversation_history,
                chat_id=payload.chat_id,
                active_document_id=payload.active_document_id,
                user_id=current_user["id"],
            ),
            media_type="text/event-stream",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Falha ao responder pergunta: {exc}"
        ) from exc


@router.post("/chat/preflight")
async def chat_preflight(
    payload: ChatRequest, current_user: dict = Depends(get_current_user)
):
    """Lightweight classification + clarifying gate without retrieval or LLM generation."""
    try:
        result = await rag_pipeline.preflight_classify(
            payload.question,
            provider=payload.provider,
            conversation_history=payload.conversation_history,
            chat_id=payload.chat_id,
            user_id=current_user["id"],
        )
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Falha na classificacao: {exc}"
        ) from exc
