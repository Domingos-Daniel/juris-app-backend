"""OpenAI-compatible chat completions endpoint for Deepgram Voice Agent.

Deepgram's Agent API can call this endpoint as the "think" provider via custom endpoint.
We translate the request → run RAG pipeline → return OpenAI-format response.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.services.rag.pipeline import rag_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


class AgentThinkRequest(BaseModel):
    messages: list[dict]
    model: str = "juris-app"


@router.post("/api/agent/think")
async def agent_think(request: Request, body: AgentThinkRequest):
    """OpenAI-compatible /v1/chat/completions endpoint for Deepgram Agent.

    Deepgram sends: POST /api/agent/think with {"messages": [...], "model": "..."}
    We run: RAG pipeline on the last user message
    Returns: OpenAI-format response
    """
    user_messages = [m for m in body.messages if m.get("role") == "user"]
    if not user_messages:
        return {"choices": [{"message": {"content": "No user message found."}}]}

    query = user_messages[-1].get("content", "").strip()
    if not query:
        return {"choices": [{"message": {"content": "Empty query."}}]}

    try:
        accumulated: list[str] = []
        async for chunk in rag_pipeline.answer_query_stream_safe(
            query=query,
            provider="deepseek",
        ):
            line = chunk.removeprefix("data: ").strip()
            if not line:
                continue
            import json as _json

            try:
                payload = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if payload.get("token"):
                accumulated.append(payload["token"])

        answer = "".join(accumulated).strip()
        if not answer:
            answer = "Desculpe, nao consegui processar a sua pergunta."

        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    except Exception as exc:
        logger.exception("Agent think error: %s", exc)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Ocorreu um erro ao processar a consulta. Tente novamente.",
                    },
                    "finish_reason": "error",
                }
            ]
        }
