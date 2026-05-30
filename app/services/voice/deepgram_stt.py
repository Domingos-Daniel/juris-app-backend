"""Real-time Speech-to-Text via Deepgram WebSocket — Nova-3, pt-PT."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import AsyncIterator

import aiohttp
from app.core.config import get_settings

logger = logging.getLogger(__name__)

DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramSTT:
    """Async lifecycle wrapper around Deepgram's real-time WebSocket STT."""

    def __init__(self) -> None:
        settings = get_settings()
        dg_key = (
            os.environ.get("DEEPGRAM_API_KEY")
            or getattr(settings, "deepseek_api_key", "")
            or ""
        )
        self._api_key: str = (dg_key or "").strip()
        self._ws = None
        self._session: aiohttp.ClientSession | None = None
        self._transcript_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._closed = False
        self._read_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Open a WebSocket to Deepgram with streaming params."""
        params = (
            "?model=nova-3"
            "&language=pt-PT"
            "&encoding=linear16"
            "&channels=1"
            "&sample_rate=24000"
            "&interim_results=true"
            "&utterance_end_ms=2000"
            "&vad_events=true"
            "&punctuate=true"
            "&smart_format=true"
        )
        url = DEEPGRAM_WS_URL + params
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(
            url,
            headers={"Authorization": f"Token {self._api_key}"},
        )
        self._read_task = asyncio.create_task(self._read_messages())
        logger.info("Deepgram WebSocket connected")

    async def _read_messages(self) -> None:
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue

                    if data.get("type") in ("SpeechStarted", "UtteranceEnd"):
                        continue

                    channel = data.get("channel") or {}
                    alternatives = channel.get("alternatives") or []
                    if not alternatives:
                        continue

                    alt = alternatives[0]
                    text = alt.get("transcript", "").strip()
                    if not text:
                        continue

                    is_final = data.get("is_final", False)
                    speech_final = data.get("speech_final", False)
                    event = {
                        "type": "final" if (is_final or speech_final) else "interim",
                        "text": text,
                        "is_final": is_final or speech_final,
                    }
                    await self._transcript_queue.put(event)

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("Deepgram WS error: %s", msg.data)
                    await self._transcript_queue.put(
                        {"type": "error", "message": str(msg.data)}
                    )
                    break
        except Exception as exc:
            logger.error("Deepgram WS read error: %s", exc)
            await self._transcript_queue.put({"type": "error", "message": str(exc)})
        finally:
            self._closed = True
            await self._transcript_queue.put({"type": "close"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws is not None and not self._closed:
            await self._ws.send_bytes(chunk)

    async def recv(self) -> dict:
        return await self._transcript_queue.get()

    async def close(self) -> None:
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._ws = None
        self._session = None
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed
