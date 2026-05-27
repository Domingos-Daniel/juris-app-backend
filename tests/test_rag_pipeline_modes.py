from app.services.rag.pipeline import rag_pipeline


def test_chat_response_model_has_answer_mode_on_no_context(monkeypatch):
    class DummyClassification:
        requires_strict_corpus_match = False
        requested_diplomas = []

        def model_dump(self):
            return {"main_branch": "administrativo"}

    class DummyRetrieval:
        retrieved_chunks = []

    async def fake_retrieve(*args, **kwargs):
        return DummyRetrieval()

    monkeypatch.setattr("app.services.rag.pipeline.legal_classifier.classify", lambda q, h: DummyClassification())
    monkeypatch.setattr("app.services.rag.pipeline.legal_retrieval_service.retrieve", fake_retrieve)

    import asyncio

    response = asyncio.run(rag_pipeline.answer_query("Qual é o custo da segunda via do bilhete de identidade?"))

    assert response.answer_mode == "refused"
    assert response.confidence["level"] == "baixa"


def test_strict_route_refuses_when_requested_diploma_is_missing_from_index(monkeypatch):
    class DummyClassification:
        requires_strict_corpus_match = True
        requested_diplomas = ["Lei das Sociedades Comerciais"]

        def model_dump(self):
            return {
                "main_branch": "comercial",
                "requested_diplomas": ["Lei das Sociedades Comerciais"],
                "requires_strict_corpus_match": True,
            }

    monkeypatch.setattr("app.services.rag.pipeline.legal_classifier.classify", lambda q, h: DummyClassification())
    monkeypatch.setattr("app.services.rag.pipeline.legislation_vector_store.available_diploma_slugs", lambda: {"codigo-civil", "lei-geral-do-trabalho-lei-12-23"})

    import asyncio

    response = asyncio.run(rag_pipeline.answer_query("Quais sao os direitos essenciais dos socios minoritarios numa sociedade por quotas?"))

    assert response.answer_mode == "refused"
    assert any(issue["code"] == "corpus_gap" for issue in response.validation_issues)
