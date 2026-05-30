"""WebSocket endpoint for real-time voice interaction with Deepgram + RAG + TTS."""

from __future__ import annotations

import asyncio
import base64
import json
import logging

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)

from app.core.auth import get_ws_current_user
from app.services.voice import DeepgramSTT
from app.services.voice.tts import generate_tts_audio
from app.services.rag.pipeline import rag_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Helpers ───────────────────────────────────────────────────────


async def _send_json(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


# ── WebSocket Handler ─────────────────────────────────────────────


@router.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket) -> None:
    """Real-time voice pipeline: Deepgram STT → RAG → TTS → browser.

    Protocol:
    - Browser sends binary audio chunks (Opus-encoded, 16kHz mono).
    - Browser sends `{"type":"done"}` to signal end of speech.
    - Server sends `{"type":"interim","text":"..."}` during speech.
    - Server sends `{"type":"transcript","text":"..."}` at end of utterance.
    - Server sends `{"type":"answer_text","text":"..."}` (streaming tokens).
    - Server sends `{"type":"audio","data":"<base64_mp3>"}`.
    - Server sends `{"type":"done"}`.
    """
    # --- Authentication -----------------------------------------------
    try:
        await ws.accept()
        # Extract token from query string
        token = ws.query_params.get("token") or ""
        user_info = get_ws_current_user(token)
        if user_info is None:
            await _send_json(ws, {"type": "error", "message": "Token invalido"})
            await ws.close(code=4001)
            return
    except (WebSocketException, Exception):
        return

    stt: DeepgramSTT | None = None
    accumulated_text: str = ""
    full_answer: list[str] = []
    uttered_final: bool = False
    transcript_sent: bool = False
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    listen_task: asyncio.Task | None = None
    process_task: asyncio.Task | None = None

    try:
        # --- Connect Deepgram ------------------------------------------
        stt = DeepgramSTT()
        await stt.connect()

        # --- Background task: feed audio to Deepgram -------------------
        async def _feed():
            try:
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        break
                    await stt.send_audio(chunk)
            except Exception as exc:
                logger.warning("Feed task error: %s", exc)

        listen_task = asyncio.create_task(_feed())

        # --- Background task: process transcription + RAG --------------
        async def _process():
            nonlocal accumulated_text, uttered_final
            full_text = ""
            while not uttered_final:
                event = await stt.recv()
                etype = event.get("type")

                if etype == "close":
                    break
                if etype == "error":
                    logger.error("Deepgram error: %s", event.get("message"))
                    await _send_json(
                        ws,
                        {
                            "type": "error",
                            "message": event.get("message", "Erro Deepgram"),
                        },
                    )
                    break

                text = event.get("text", "")
                if etype == "interim":
                    await _send_json(ws, {"type": "interim", "text": text})
                elif etype == "final":
                    full_text += (" " if full_text else "") + text
                    accumulated_text = full_text
                    uttered_final = True
                    transcript_sent = True
                    await _send_json(ws, {"type": "transcript", "text": full_text})
                    break

        process_task = asyncio.create_task(_process())

        # --- Main loop: receive audio from browser --------------------
        while True:
            try:
                data = await ws.receive()
            except WebSocketDisconnect:
                break

            if "bytes" in data:
                await audio_queue.put(data["bytes"])
            elif "text" in data:
                try:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "done":
                        uttered_final = True
                        break
                except json.JSONDecodeError:
                    pass

        # --- Wait for transcription to finalize ------------------------
        try:
            if process_task:
                await asyncio.wait_for(process_task, timeout=15)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for Deepgram finalization")

        # --- Voice flow: only STT → send transcript → done ---
        # The chat response (RAG) and TTS are handled by the normal SSE chat flow
        # and the /api/tts/speak endpoint, avoiding duplicate RAG calls and ensuring
        # the spoken answer matches the displayed answer.
        if (
            not transcript_sent
            and accumulated_text.strip()
            and accumulated_text.strip() != "."
        ):
            await _send_json(
                ws, {"type": "transcript", "text": accumulated_text.strip()}
            )

        await _send_json(ws, {"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Voice WebSocket error: %s", exc)
        try:
            await _send_json(ws, {"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if listen_task:
            listen_task.cancel()
        if process_task:
            process_task.cancel()
        if stt:
            await stt.close()
        try:
            await ws.close()
        except Exception:
            pass
