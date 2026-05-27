import pytest
pytestmark = pytest.mark.asyncio
from unittest.mock import AsyncMock

import pytest

from app.db.models import RetrievedChunk
from app.services.legal.models import LLMAnswerDraft, RetrievalEvidence, RetrievalResult
from app.services.rag import pipeline as rag_pipeline_module


pytestmark = pytest.mark.asyncio


class DummyChatStore:
    def __init__(self) -> None:
        self.saved = []

    def create_chat(self, title: str, active_document_id=None, user_id=None):
        return "chat-test"

    def append_chat_exchange(self, **kwargs):
        self.saved.append(kwargs)

    def save_query(self, question: str, answer: str):
        self.saved.append({"question": question, "answer": answer})


def _chunk(title: str, text: str, article_number: str | None, branch: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{title}-1",
        text=text,
        source=title,
        title=title,
        link_original="https://example.test/doc.pdf",
        page=7,
        article_number=article_number,
        law_status="Vigente",
        distance=0.1,
        source_scope="official",
        document_id=None,
        metadata={"legal_branch": branch, "is_normative": True, "normative_density": 5.0},
    )


async async def test_pipeline_returns_structured_metadata(monkeypatch):
    classification = rag_pipeline_module.await legal_classifier.classify("Fui despedido sem aviso e sem indemnização. O que faço agora?")
    chunk = _chunk("Lei Geral do Trabalho (Lei 12/23)", "Artigo 12 O trabalhador tem direito à compensação quando...", "12", "laboral")
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 9.5, "base", "official")],
        retrieved_chunks=[chunk],
    )
    chat_store = DummyChatStore()

    monkeypatch.setattr(rag_pipeline_module, "postgres_manager", chat_store)
    monkeypatch.setattr(rag_pipeline_module.legal_retrieval_service, "retrieve", AsyncMock(return_value=retrieval))
    monkeypatch.setattr(rag_pipeline_module.llm_router, "generate", AsyncMock(return_value=(
        '{"direct_answer":"Há indícios de ilicitude laboral a confirmar pelo contexto.","simple_explanation":"Se foi despedido sem seguir as regras, pode reagir.","technical_analysis":"A questão centra-se na regularidade da cessação.","practical_steps":["Guardar contrato e comunicações.","Reunir recibos e prova do despedimento."],"distinctions":["Compensação e indemnização não são necessariamente sinónimos."],"prudent_inferences":["A resposta depende dos factos exactos da cessação."],"additional_validation_needed":["Convém confirmar os artigos exactos aplicáveis ao caso concreto."],"cited_articles":["12"],"cited_diplomas":["Lei Geral do Trabalho"]}',
        "openai",
    )))

    response = await rag_pipeline_module.rag_pipeline.answer_query(
        "Fui despedido sem aviso e sem indemnização. O que faço agora?",
        provider="openai",
        conversation_history=[],
        chat_id=None,
        active_document_id=None,
        user_id="u1",
    )

    assert response.provider_used == "openai"
    assert response.classification is not None
    assert response.confidence is not None
    assert response.legal_basis
    assert "Base legal" in response.answer
    assert "Confiança da resposta" in response.answer


async async def test_pipeline_filters_llm_articles_outside_context(monkeypatch):
    classification = rag_pipeline_module.await legal_classifier.classify("Fui despedido sem aviso e sem indemnização. O que faço agora?")
    chunk = _chunk("Lei Geral do Trabalho (Lei 12/23)", "Artigo 12 O trabalhador tem direito à compensação quando...", "12", "laboral")
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 9.5, "base", "official")],
        retrieved_chunks=[chunk],
    )
    chat_store = DummyChatStore()

    monkeypatch.setattr(rag_pipeline_module, "postgres_manager", chat_store)
    monkeypatch.setattr(rag_pipeline_module.legal_retrieval_service, "retrieve", AsyncMock(return_value=retrieval))
    monkeypatch.setattr(rag_pipeline_module.llm_router, "generate", AsyncMock(return_value=(
        '{"direct_answer":"Há indícios de ilicitude laboral a confirmar pelo contexto.","simple_explanation":"Se foi despedido sem seguir as regras, pode reagir.","technical_analysis":"A questão centra-se na regularidade da cessação.","practical_steps":["Guardar contrato e comunicações."],"distinctions":[],"prudent_inferences":[],"additional_validation_needed":[],"cited_articles":["99"],"cited_diplomas":["Lei Geral do Trabalho"]}',
        "openai",
    )))

    response = await rag_pipeline_module.rag_pipeline.answer_query(
        "Fui despedido sem aviso e sem indemnização. O que faço agora?",
        provider="openai",
        conversation_history=[],
        chat_id=None,
        active_document_id=None,
        user_id="u1",
    )

    assert not any(issue["code"] == "unsupported_article" for issue in response.validation_issues)
    assert response.confidence["score"] >= 0.62
    assert response.confidence["level"] in {"media", "alta"}
    assert any(item["article"] == "12" for item in response.legal_basis)
