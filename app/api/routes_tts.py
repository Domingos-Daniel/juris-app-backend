"""Simple REST endpoint for Text-to-Speech. Used by frontend after chat response."""

from __future__ import annotations

import logging
from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.services.voice.tts import generate_tts_audio

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/tts/speak")
async def tts_speak(request: Request) -> Response:
    """Generate MP3 audio from text. POST body is plain text (UTF-8)."""
    body = await request.body()
    text = body.decode("utf-8").strip()
    if not text:
        return Response(status_code=400)

    text = text[:800]  # limit for TTS
    audio = await generate_tts_audio(text)
    return Response(content=audio, media_type="audio/mpeg")
