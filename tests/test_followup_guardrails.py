from app.db.models import RetrievedChunk
from app.services.legal.models import (
    LegalClassification,
    RetrievalEvidence,
    RetrievalResult,
)
from app.services.legal.normative_guardrails import normative_guardrails_service
from app.services.rag.pipeline import (
    RAGPipeline,
    _needs_clarification_from_classification,
)


def _classification(**overrides) -> LegalClassification:
    base = {
        "query_text": "",
        "main_branch": "constitucional",
        "branch_candidates": ["constitucional"],
        "request_type": "explicacao_simples",
        "specificity": "follow_up",
        "audience": "leigo",
        "is_follow_up": True,
        "topic_route": "constitucional",
        "search_query": "",
    }
    base.update(overrides)
    return LegalClassification(**base)


def _evidence(chunk: RetrievedChunk, score: float = 5.0) -> RetrievalEvidence:
    return RetrievalEvidence(
        query_used="teste",
        chunk=chunk,
        score=score,
        retrieval_reason="base",
        source_bucket="official",
    )


def test_followup_hydration_keeps_original_article_anchor():
    history = [
        "Utilizador: O que diz o artigo 26 da Constituição da República de Angola?",
        "Assistente: Não consegui confirmar esse artigo no contexto recuperado.",
    ]
    classification = _classification()
    hydrated = RAGPipeline._hydrate_follow_up_context(
        "Esse mesmo artigo fala de quê?",
        history,
        classification,
        {
            "active_article": "26",
            "diploma_slug": "constituicao-republica-angola-2022",
            "metadata": {
                "unresolved_requested_article": "26",
                "last_requested_diploma": "Constituição da República de Angola",
            },
        },
    )

    assert hydrated.requested_article_numbers == ["26"]
    assert "Constituição da República de Angola" in hydrated.requested_diplomas
    assert hydrated.needs_article_validation is True
    assert hydrated.requires_strict_corpus_match is True


def test_normative_guardrails_flag_vigency_and_conflict():
    classification = _classification(
        query_text="Essa lei ainda está em vigor?",
        requested_diplomas=["Código Civil"],
        requested_article_numbers=["26"],
        needs_article_validation=True,
        requires_strict_corpus_match=True,
        specificity="validacao_base_legal",
        is_follow_up=False,
    )
    chunk_a = RetrievedChunk(
        chunk_id="a",
        text="Artigo 26. Texto principal sem confirmação expressa de vigência.",
        source="Documento A",
        title="Documento A",
        link_original=None,
        page=10,
        article_number="26",
        law_status="Nao verificado",
        source_scope="official",
        document_id="doc-a",
        metadata={
            "article_main": "26",
            "article_references": ["26"],
            "diploma_slug": "codigo-civil",
            "legal_branch": "civil",
        },
    )
    chunk_b = RetrievedChunk(
        chunk_id="b",
        text="Artigo 26. Norma posterior que altera disposições anteriores.",
        source="Documento B",
        title="Documento B",
        link_original=None,
        page=8,
        article_number="26",
        law_status="Nao verificado",
        source_scope="official",
        document_id="doc-b",
        metadata={
            "article_main": "26",
            "article_references": ["26"],
            "diploma_slug": "lei-especial-x",
            "legal_branch": "civil",
        },
    )
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[_evidence(chunk_a), _evidence(chunk_b, score=4.9)],
        retrieved_chunks=[chunk_a, chunk_b],
    )

    normative_status, _notes, issues, jurisprudence_basis = (
        normative_guardrails_service.analyze(classification, retrieval, [])
    )
    issue_codes = {issue.code for issue in issues}

    assert normative_status == "partially_known"
    assert "vigency_unverified" in issue_codes
    assert "normative_conflict" in issue_codes
    assert jurisprudence_basis == []


def test_followup_context_does_not_force_vague_gate():
    classification = _classification(
        query_text="Fale mais",
        main_branch="indeterminado",
        topic_route="geral",
        is_follow_up=True,
        specificity="follow_up",
        semantic_confidence=0.05,
    )

    needs = _needs_clarification_from_classification(
        classification,
        semantic_confidence=classification.semantic_confidence,
        has_follow_up_context=True,
    )
    assert needs is False
