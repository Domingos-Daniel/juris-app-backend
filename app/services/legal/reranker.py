from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class LocalReranker:
    """
    Reranker local baseado em CrossEncoder (sentence-transformers).

    Substitui o reranker LLM anterior que fazia N chamadas HTTP por chunk.
    Este roda inteiramente na CPU local, sem latência de rede.

    Modelo padrão: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Multilíngue suficiente para Português/Angolano em contexto jurídico
    - Rápido (~20ms por par em CPU)
    - Já incluído no sentence-transformers que está instalado
    """

    _MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _RELEVANCE_THRESHOLD = 0.0  # score logit — qualquer valor > 0 é relevante

    def __init__(self) -> None:
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._MODEL_NAME)
                logger.info("LocalReranker: CrossEncoder carregado com sucesso (%s)", self._MODEL_NAME)
            except Exception as exc:
                logger.warning(
                    "LocalReranker: falha ao carregar CrossEncoder (%s). "
                    "Reranking desactivado — todos os chunks aceites. Erro: %s",
                    self._MODEL_NAME, exc,
                )
                self._model = None
        return self._model

    async def rerank(
        self,
        question: str,
        chunks_text: list[str],
        provider: str | None = None,  # ignorado — mantido por compatibilidade de assinatura
    ) -> list[bool]:
        """
        Retorna lista de booleanos indicando se cada chunk é relevante.

        - Se o modelo não estiver disponível, aceita todos os chunks.
        - Trunca cada chunk a 500 chars para eficiência.
        - Garante que pelo menos 1 chunk é aceite (fallback).
        """
        if len(chunks_text) <= 2:
            return [True] * len(chunks_text)

        model = self.model
        if model is None:
            # Fallback seguro: aceitar todos
            return [True] * len(chunks_text)

        pairs = [(question, text[:500]) for text in chunks_text]

        try:
            scores: list[float] = await asyncio.to_thread(model.predict, pairs)
        except Exception as exc:
            logger.warning("LocalReranker: erro ao calcular scores: %s. Aceitando todos.", exc)
            return [True] * len(chunks_text)

        results = [float(score) > self._RELEVANCE_THRESHOLD for score in scores]

        # Garantia: sempre pelo menos 1 chunk aceite
        if not any(results):
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            results[best_idx] = True

        accepted = sum(results)
        logger.debug(
            "LocalReranker: %d/%d chunks aceites (threshold=%.2f)",
            accepted, len(chunks_text), self._RELEVANCE_THRESHOLD,
        )
        return results


llm_reranker = LocalReranker()
