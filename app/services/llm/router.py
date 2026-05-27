from __future__ import annotations

import asyncio
import random
from urllib.error import HTTPError

from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.llm.deepseek_client import deepseek_client
from app.services.llm.openai_client import openai_client
from app.services.llm.openrouter_client import openrouter_client
from app.services.llm.opencode import opencode_client

log = get_logger(__name__)

RETRYABLE_STATUSES = {429, 502, 503, 504}
MAX_RETRIES = 3


class LLMRouter:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def generate(
        self,
        prompt: str,
        provider: str | None = None,
        system_prompt: str | None = None,
        json_mode: bool = True,
        max_tokens: int = 2000,
    ) -> tuple[str, str]:
        selected_provider = (
            provider or self.settings.default_llm_provider or "deepseek"
        ).lower()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await self._call_provider(
                    selected_provider,
                    prompt,
                    system_prompt=system_prompt,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                )
            except HTTPError as exc:
                if exc.code in RETRYABLE_STATUSES and attempt < MAX_RETRIES:
                    delay = (2**attempt) + random.uniform(0, 1)
                    log.warning(
                        "LLM %s retry %d/%d after HTTP %d (%.1fs delay)",
                        selected_provider,
                        attempt,
                        MAX_RETRIES,
                        exc.code,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(
                    f"API do modelo {selected_provider} indisponível "
                    f"(HTTP {exc.code}). Tente novamente dentro de alguns segundos."
                ) from exc

    async def _call_provider(
        self,
        selected_provider: str,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = True,
        max_tokens: int = 2000,
    ) -> tuple[str, str]:
        if selected_provider == "deepseek":
            answer = await deepseek_client.generate(
                prompt,
                system_prompt=system_prompt,
                json_mode=json_mode,
                max_tokens=max_tokens,
            )
            return answer, "deepseek"

        if selected_provider == "openrouter":
            answer = await openrouter_client.generate(
                prompt,
                system_prompt=system_prompt,
                json_mode=json_mode,
                max_tokens=max_tokens,
            )
            return answer, "openrouter"

        if selected_provider == "openai":
            answer = await openai_client.generate(
                prompt,
                system_prompt=system_prompt,
                json_mode=json_mode,
                max_tokens=max_tokens,
            )
            return answer, "openai"

        if selected_provider == "opencode":
            answer = await opencode_client.generate(
                prompt,
                system_prompt=system_prompt,
                json_mode=json_mode,
                max_tokens=max_tokens,
            )
            return answer, "opencode"

        answer = await openrouter_client.generate(
            prompt,
            system_prompt=system_prompt,
            json_mode=json_mode,
            max_tokens=max_tokens,
        )
        return answer, "openrouter"


llm_router = LLMRouter()
