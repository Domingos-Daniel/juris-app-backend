from __future__ import annotations

import asyncio
import json

import requests

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class OpenCodeClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = True,
        max_tokens: int = 700,
    ) -> str:
        if not self.settings.opencode_api_key:
            raise RuntimeError("OPENCODE_API_KEY nao configurada")

        sys_content = (
            system_prompt
            or "Es um assistente juridico focado na legislacao angolana. Responde apenas com base no contexto fornecido. Quando o pedido exigir JSON, devolve apenas JSON valido."
        )

        payload = {
            "model": self.settings.opencode_model,
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        data = await asyncio.to_thread(self._request_completion, payload)

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"Resposta vazia do OpenCode. Raw: {json.dumps(data, ensure_ascii=False)[:500]}"
            )
        message = choices[0].get("message") or {}
        finish_reason = choices[0].get("finish_reason", "")
        content = message.get("content")
        if isinstance(content, list):
            text_parts = [
                part.get("text", "") for part in content if isinstance(part, dict)
            ]
            content = "\n".join(part for part in text_parts if part)

        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if not content and reasoning:
            logger.warning(
                "OpenCode: modelo devolveu apenas reasoning_content (%d chars), content vazio. "
                "finish_reason=%s. Retornando vazio para fallback do pipeline.",
                len(str(reasoning)),
                finish_reason,
            )
            return ""

        if not content:
            raise RuntimeError(
                f"Campo content ausente na resposta do OpenCode. "
                f"keys={list(data.keys())} finish_reason={finish_reason} "
                f"message_keys={list(message.keys())}"
            )

        return str(content).strip()

    def _request_completion(self, payload: dict) -> dict:
        import time as time_module

        max_retries = 5
        for attempt in range(max_retries):
            resp = requests.post(
                f"{self.settings.opencode_base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.settings.opencode_api_key}",
                },
                timeout=self.settings.request_timeout_seconds,
            )
            if resp.status_code == 429 and attempt < max_retries - 1:
                wait = 2**attempt * 5
                time_module.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()


opencode_client = OpenCodeClient()
