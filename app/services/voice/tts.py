"""Text-to-Speech via Microsoft Edge TTS (gratuito, PT-PT natural).

Uses the edge-tts Python module directly (no subprocess).
"""

from __future__ import annotations

import asyncio
import io
import logging

logger = logging.getLogger(__name__)

_PT_VOICES = [
    "pt-PT-DuarteNeural",
    "pt-PT-RaquelNeural",
]


async def generate_tts_audio(
    text: str, voice: str | None = None, fmt: str = "mp3"
) -> bytes:
    """Generate speech audio from text using Microsoft Edge TTS.

    Uses edge-tts Communicate API (async, no subprocess needed).
    Works on Windows, Linux, macOS.

    Args:
        text: The text to speak (plain text, max ~500 chars).
        voice: 'pt-PT-DuarteNeural', 'pt-PT-FernandaNeural', 'pt-PT-RaquelNeural'.
        fmt: Output format — only 'mp3' supported.

    Returns:
        Raw audio bytes (MP3 format).
    """
    if not voice:
        voice = _PT_VOICES[1]  # Raquel (female)
    if voice not in _PT_VOICES:
        voice = _PT_VOICES[1]

    clipped = text.strip()[:800]
    if not clipped:
        return b""

    try:
        import edge_tts

        communicate = edge_tts.Communicate(clipped, voice)
        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        return audio_buffer.getvalue()
    except ImportError:
        logger.warning("edge-tts not installed — TTS disabled")
        return b""


async def generate_tts_stream(text: str, voice: str | None = None):
    """Stream audio chunks (for future use with longer texts)."""
    audio_bytes = await generate_tts_audio(text, voice=voice)
    chunk_size = 4096
    for i in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[i : i + chunk_size]
