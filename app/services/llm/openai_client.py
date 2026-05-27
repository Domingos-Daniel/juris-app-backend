from __future__ import annotations

import asyncio
import json
import urllib.request

from app.core.config import get_settings
from app.services.llm.http_transport import read_json_response


class OpenAIClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def generate(self, prompt: str, system_prompt: str | None = None, json_mode: bool = True, max_tokens: int = 700) -> str:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY nao configurada")

        sys_content = system_prompt or "És um assistente jurídico focado na legislação angolana. Responde apenas com base no contexto fornecido, em português europeu correcto, com linguagem natural, profissional e clara. Não uses placeholders, notas técnicas internas nem formulações artificiais. Quando o pedido exigir JSON, devolve apenas JSON válido."

        payload = {
            "model": self.settings.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": sys_content,
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = await asyncio.to_thread(self._request_completion, payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Resposta vazia da OpenAI")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            content = "\n".join(part for part in text_parts if part)
        if not content:
            raise RuntimeError("Campo content ausente na resposta da OpenAI")
        return str(content).strip()

    def _request_completion(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.settings.openai_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
        )
        return read_json_response(request, timeout=self.settings.request_timeout_seconds)


openai_client = OpenAIClient()
