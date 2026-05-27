from __future__ import annotations

import asyncio
import json
import urllib.request

from app.core.config import get_settings
from app.services.llm.http_transport import read_json_response


class OpenAIEmbeddingClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY nao configurada para embeddings")
        return await asyncio.to_thread(self._request_embeddings, texts)

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.settings.openai_embedding_model,
            "input": texts,
        }
        request = urllib.request.Request(
            f"{self.settings.openai_base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
        )
        data = read_json_response(request, timeout=self.settings.request_timeout_seconds)
        return [item["embedding"] for item in data.get("data", [])]


openai_embedding_client = OpenAIEmbeddingClient()
