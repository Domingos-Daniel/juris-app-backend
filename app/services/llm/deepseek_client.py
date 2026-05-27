from __future__ import annotations

import asyncio
import json
import logging
import urllib.request

import httpx
from app.core.config import get_settings

logger = logging.getLogger(__name__)
from app.services.llm.http_transport import (
    read_json_response,
    urlopen_without_env_proxy,
)


class DeepSeekClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._http_client: httpx.AsyncClient | None = None

    def set_http_client(self, client: httpx.AsyncClient) -> None:
        self._http_client = client

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = True,
        max_tokens: int = 700,
    ) -> str:
        if not self.settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY nao configurada")

        sys_content = (
            system_prompt
            or "Es um assistente juridico focado na legislacao angolana. Responde apenas com base no contexto fornecido, em portugues correcto, com linguagem natural, profissional e clara. Quando o pedido exigir JSON, devolve apenas JSON valido."
        )

        payload = self._build_payload(sys_content, prompt, json_mode, max_tokens)
        data = await asyncio.to_thread(self._request_completion, payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"Resposta vazia do DeepSeek. Raw: {json.dumps(data, ensure_ascii=False)[:500]}"
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            text_parts = [
                part.get("text", "") for part in content if isinstance(part, dict)
            ]
            content = "\n".join(part for part in text_parts if part)
        if not content:
            raise RuntimeError(
                f"Campo content ausente na resposta do DeepSeek. "
                f"finish_reason={choices[0].get('finish_reason')} "
                f"message_keys={list(message.keys())}"
            )
        return str(content).strip()

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = True,
        max_tokens: int = 700,
    ):
        """Stream tokens from DeepSeek API. Yields each content delta string."""
        if not self.settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY nao configurada")
        sys_content = (
            system_prompt
            or "Es um assistente juridico focado na legislacao angolana. Responde apenas com base no contexto fornecido, em portugues correcto, com linguagem natural, profissional e clara. Quando o pedido exigir JSON, devolve apenas JSON valido."
        )
        payload = self._build_payload(sys_content, prompt, json_mode, max_tokens)
        payload["stream"] = True

        full_content: list[str] = []
        async for token in self._request_stream(payload):
            full_content.append(token)
            yield token

    async def _request_stream(self, payload: dict):
        """Read SSE stream from DeepSeek and yield content deltas.

        Uses the shared httpx.AsyncClient (set via set_http_client) for connection
        pooling — avoids creating a new TCP connection per stream.
        """
        client = self._http_client
        if client is None:
            import httpx

            client = httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
            async with client:
                async for token in self._read_sse(client, payload):
                    yield token
        else:
            async for token in self._read_sse(client, payload):
                yield token

    async def _read_sse(self, client, payload: dict):
        async with client.stream(
            "POST",
            "https://api.deepseek.com/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(
                    f"DeepSeek stream error {resp.status_code}: {body.decode()[:500]}"
                )
            logger.debug("DeepSeek stream connected, reading SSE lines...")
            line_count = 0
            async for line in resp.aiter_lines():
                line_count += 1
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                if not line_stripped.startswith("data: "):
                    logger.debug("Unexpected SSE line: %s", line_stripped[:200])
                    continue
                data_str = line_stripped[6:]
                if data_str == "[DONE]":
                    logger.debug("SSE stream complete: %d lines", line_count)
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

            if line_count == 0:
                logger.warning("DeepSeek SSE stream returned no lines")

    def _build_payload(
        self, sys_content: str, prompt: str, json_mode: bool, max_tokens: int
    ) -> dict:
        payload = {
            "model": self.settings.deepseek_model,
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _request_completion(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"https://api.deepseek.com/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
        )
        return read_json_response(
            request, timeout=self.settings.request_timeout_seconds
        )


deepseek_client = DeepSeekClient()
