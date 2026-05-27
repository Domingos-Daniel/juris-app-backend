import pytest
pytestmark = pytest.mark.asyncio
from app.db.models import RetrievedChunk
from app.services.legal.classification import legal_classifier
from app.services.legal.composition import legal_composer
from app.services.legal.confidence import legal_confidence_service
from app.services.legal.models import LLMAnswerDraft, RetrievalEvidence, RetrievalResult
from app.services.legal.validation import legal_validation_service


def _chunk(title: str, text: str, article_number: str | None = None, source_scope: str = "official", branch: str = "laboral") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{title}-1",
        text=text,
        source=title,
        title=title,
        link_original="https://example.test/doc.pdf",
        page=10,
        article_number=article_number,
        law_status="Vigente",
        distance=0.2,
        source_scope=source_scope,
        document_id=None,
        metadata={"legal_branch": branch, "is_normative": True, "normative_density": 4.0},
    )


async def test_flags_unsupported_articles():
    classification = await legal_classifier.classify("Qual o artigo aplicável ao despedimento sem aviso?")
    chunk = _chunk("Lei Geral do Trabalho (Lei 12/23)", "Artigo 12 O trabalhador tem direito ...", "12")
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 8.0, "base", "official")],
        retrieved_chunks=[chunk],
    )
    draft = LLMAnswerDraft(
        direct_answer="Resposta curta",
        cited_articles=["99"],
        cited_diplomas=["Lei Geral do Trabalho"],
    )

    validation = legal_validation_service.validate(classification, retrieval, draft)

    assert validation.unsupported_articles == ["99"]
    assert any(issue.code == "unsupported_article" for issue in validation.issues)


async def test_promulgation_chunk_is_not_treated_as_confirmed_article_support():
    classification = await legal_classifier.classify("Qual é o prazo de recurso no CPP para prisão preventiva?")
    chunk = RetrievedChunk(
        chunk_id="cpp-front-1",
        text="Artigo 10.º da Lei n.º 11/75 ... Aprova o Código do Processo Penal Angolano e revoga disposições legais anteriores.",
        source="Codigo-Processo-Penal-Lei-39-20.pdf",
        title="Codigo do Processo Penal (Lei 39/20)",
        link_original="https://example.test/cpp.pdf",
        page=1,
        article_number="10, 4, 33, 14",
        law_status="Vigente",
        distance=0.1,
        source_scope="official",
        document_id=None,
        metadata={
            "legal_branch": "penal",
            "diploma_slug": "codigo-processo-penal-lei-39-20",
            "segmentation": "article_block",
            "article_main": "10",
            "article_references": ["10", "4", "33", "14"],
            "is_normative": True,
            "normative_density": 3.0,
        },
    )
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 8.0, "topic_route", "official")],
        retrieved_chunks=[chunk],
    )
    draft = LLMAnswerDraft(
        direct_answer="O prazo é de cinco dias.",
        cited_articles=["10"],
        cited_diplomas=["Código do Processo Penal"],
    )

    validation = legal_validation_service.validate(classification, retrieval, draft)

    assert validation.confirmed_legal_basis == []
    assert any(issue.code in {"strict_confirmation_gap", "weak_article_confirmation", "limited_contextual_support"} for issue in validation.issues)


async def test_detects_mixed_sources():
    classification = await legal_classifier.classify("Analise o contrato do utilizador e compare com a lei laboral")
    official_chunk = _chunk("Lei Geral do Trabalho (Lei 12/23)", "Artigo 20 O empregador ...", "20", "official", "laboral")
    user_chunk = _chunk("Contrato.pdf", "O contrato assinado prevê ...", None, "user_upload", "laboral")
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", official_chunk, 9.0, "base", "official")],
        user_evidence=[RetrievalEvidence("q", user_chunk, 5.0, "active_document", "user_upload")],
        retrieved_chunks=[official_chunk, user_chunk],
    )
    draft = LLMAnswerDraft(direct_answer="Resposta", cited_articles=["20"])

    validation = legal_validation_service.validate(classification, retrieval, draft)

    assert validation.source_cross_contamination is True
    assert any(issue.code == "mixed_sources" for issue in validation.issues)


async def test_strict_question_without_confirmed_article_is_limited_mode():
    classification = await legal_classifier.classify("Qual é o prazo de recurso no CPP para prisão preventiva?")
    chunk = RetrievedChunk(
        chunk_id="cpp-front-2",
        text="Artigo 10.º da Lei n.º 11/75 aprova o código e revoga disposições legais anteriores.",
        source="Codigo-Processo-Penal-Lei-39-20.pdf",
        title="Codigo do Processo Penal (Lei 39/20)",
        link_original="https://example.test/cpp.pdf",
        page=1,
        article_number="10, 4, 33",
        law_status="Vigente",
        distance=0.1,
        source_scope="official",
        document_id=None,
        metadata={
            "legal_branch": "penal",
            "diploma_slug": "codigo-processo-penal-lei-39-20",
            "segmentation": "article_block",
            "article_main": "10",
            "article_references": ["10", "4", "33"],
            "is_normative": True,
            "normative_density": 2.0,
        },
    )
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 8.0, "topic_route", "official")],
        retrieved_chunks=[chunk],
    )
    draft = LLMAnswerDraft(direct_answer="O prazo é de cinco dias.", cited_articles=["10"])

    validation = legal_validation_service.validate(classification, retrieval, draft)

    assert validation.answer_mode == "limited"


async def test_confidence_is_never_high_for_limited_mode():
    classification = await legal_classifier.classify("Qual é o prazo de recurso no CPP para prisão preventiva?")
    chunk = RetrievedChunk(
        chunk_id="cpp-front-3",
        text="Artigo 10.º da Lei n.º 11/75 aprova o código e revoga disposições legais anteriores.",
        source="Codigo-Processo-Penal-Lei-39-20.pdf",
        title="Codigo do Processo Penal (Lei 39/20)",
        link_original="https://example.test/cpp.pdf",
        page=1,
        article_number="10, 4, 33",
        law_status="Vigente",
        distance=0.1,
        source_scope="official",
        document_id=None,
        metadata={
            "legal_branch": "penal",
            "diploma_slug": "codigo-processo-penal-lei-39-20",
            "segmentation": "article_block",
            "article_main": "10",
            "article_references": ["10", "4", "33"],
            "is_normative": True,
            "normative_density": 2.0,
        },
    )
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 8.0, "topic_route", "official")],
        retrieved_chunks=[chunk],
    )
    draft = LLMAnswerDraft(direct_answer="O prazo é de cinco dias.", cited_articles=["10"])

    validation = legal_validation_service.validate(classification, retrieval, draft)
    confidence = legal_confidence_service.score(classification, retrieval, validation)

    assert validation.answer_mode == "limited"
    assert confidence.level != "alta"


async def test_mixed_labor_penal_without_material_penal_support_is_limited():
    classification = await legal_classifier.classify(
        "Quando um empregador deixa de pagar valores ao trabalhador, como distinguir conflito laboral de relevância penal?"
    )
    penal_chunk = RetrievedChunk(
        chunk_id="cpp-364",
        text="ARTIGO 364.º (Participação económica em negócio) O funcionário que, com intenção de obter vantagem que não seja devida, participar em negócio jurídico...",
        source="Codigo-Penal.pdf",
        title="Codigo Penal (Lei 38/20)",
        link_original="https://example.test/cp.pdf",
        page=56,
        article_number="364",
        law_status="Vigente",
        distance=0.1,
        source_scope="official",
        document_id=None,
        metadata={
            "legal_branch": "penal",
            "diploma_slug": "codigo-penal-lei-38-20",
            "segmentation": "article_block",
            "article_main": "364",
            "article_references": ["364"],
            "is_normative": True,
            "normative_density": 4.0,
        },
    )
    labor_chunk = RetrievedChunk(
        chunk_id="lgt-247",
        text="ARTIGO 247.º (Documento de pagamento) O pagamento do salário é comprovado por recibo assinado pelo trabalhador...",
        source="Lei-Geral-do-Trabalho.pdf",
        title="Lei Geral do Trabalho (Lei 12/23)",
        link_original="https://example.test/lgt.pdf",
        page=131,
        article_number="247",
        law_status="Vigente",
        distance=0.1,
        source_scope="official",
        document_id=None,
        metadata={
            "legal_branch": "laboral",
            "diploma_slug": "lei-geral-do-trabalho-lei-12-23",
            "segmentation": "article_block",
            "article_main": "247",
            "article_references": ["247"],
            "is_normative": True,
            "normative_density": 4.0,
        },
    )
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[
            RetrievalEvidence("q", penal_chunk, 9.0, "base", "official"),
            RetrievalEvidence("q", labor_chunk, 8.5, "base", "official"),
        ],
        retrieved_chunks=[penal_chunk, labor_chunk],
    )
    draft = LLMAnswerDraft(
        direct_answer="Pode haver conflito laboral e, em alguns casos, relevância penal.",
        simple_explanation="O não pagamento de valores ao trabalhador pode exigir distinguir incumprimento laboral de crime patrimonial.",
        cited_articles=["364", "247"],
    )

    validation = legal_validation_service.validate(classification, retrieval, draft)
    confidence = legal_confidence_service.score(classification, retrieval, validation)

    assert any(issue.code == "penal_relevance_gap" for issue in validation.issues)
    assert validation.answer_mode == "limited"
    assert confidence.level != "alta"


async def test_cpp_recurso_question_needs_specific_processual_support():
    classification = await legal_classifier.classify("Qual é o prazo de recurso no CPP para prisão preventiva?")
    chunk = RetrievedChunk(
        chunk_id="cpp-283",
        text="ARTIGO 283.º (Prazos máximos de prisão preventiva) A prisão preventiva cessa quando, desde o seu início, decorrerem...",
        source="Codigo-Processo-Penal.pdf",
        title="Codigo do Processo Penal (Lei 39/20)",
        link_original="https://example.test/cpp.pdf",
        page=122,
        article_number="283",
        law_status="Vigente",
        distance=0.1,
        source_scope="official",
        document_id=None,
        metadata={
            "legal_branch": "penal",
            "diploma_slug": "codigo-processo-penal-lei-39-20",
            "segmentation": "article_block",
            "article_main": "283",
            "article_references": ["283"],
            "is_normative": True,
            "normative_density": 4.0,
        },
    )
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 9.0, "topic_route", "official")],
        retrieved_chunks=[chunk],
    )
    draft = LLMAnswerDraft(
        direct_answer="O prazo de recurso é de 10 dias.",
        simple_explanation="Após a decisão que determina a prisão preventiva, o arguido tem 10 dias para recorrer.",
        cited_articles=["283"],
    )

    validation = legal_validation_service.validate(classification, retrieval, draft)
    confidence = legal_confidence_service.score(classification, retrieval, validation)

    assert any(issue.code == "processual_specificity_gap" for issue in validation.issues)
    assert validation.answer_mode == "limited"
    assert confidence.level != "alta"


async def test_cpp_confirmed_articles_must_match_query_semantics_for_grounding():
    classification = await legal_classifier.classify("Qual é o prazo de recurso no CPP para prisão preventiva?")
    chunk = RetrievedChunk(
        chunk_id="cpp-276",
        text="ARTIGO 276.º (Aplicação da medida) O juiz pode impor medida de coacção e os prazos seguem o artigo 283.º para a prisão preventiva.",
        source="Codigo-Processo-Penal.pdf",
        title="Codigo do Processo Penal (Lei 39/20)",
        link_original="https://example.test/cpp.pdf",
        page=121,
        article_number="276, 283",
        law_status="Vigente",
        distance=0.1,
        source_scope="official",
        document_id=None,
        metadata={
            "legal_branch": "penal",
            "diploma_slug": "codigo-processo-penal-lei-39-20",
            "segmentation": "article_block",
            "article_main": "276",
            "article_references": ["276", "283"],
            "is_normative": True,
            "normative_density": 4.0,
        },
    )
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 9.0, "topic_route", "official")],
        retrieved_chunks=[chunk],
    )
    draft = LLMAnswerDraft(
        direct_answer="O prazo de recurso é de 10 dias.",
        simple_explanation="Depois da decisão de prisão preventiva, o arguido pode recorrer.",
        cited_articles=["276"],
    )

    validation = legal_validation_service.validate(classification, retrieval, draft)

    assert any(issue.code == "processual_specificity_gap" for issue in validation.issues)
    assert validation.answer_mode == "limited"


async def test_cpp_numeric_claim_without_context_number_is_limited():
    classification = await legal_classifier.classify("Qual é o prazo de recurso no CPP para prisão preventiva?")
    chunk = RetrievedChunk(
        chunk_id="cpp-268",
        text="ARTIGO 268.º (Extinção das medidas de coacção) A sentença absolutória e o recurso podem extinguir medidas de prisão preventiva em certos casos.",
        source="Codigo-Processo-Penal.pdf",
        title="Codigo do Processo Penal (Lei 39/20)",
        link_original="https://example.test/cpp.pdf",
        page=120,
        article_number="268",
        law_status="Vigente",
        distance=0.1,
        source_scope="official",
        document_id=None,
        metadata={
            "legal_branch": "penal",
            "diploma_slug": "codigo-processo-penal-lei-39-20",
            "segmentation": "article_block",
            "article_main": "268",
            "article_references": ["268"],
            "is_normative": True,
            "normative_density": 4.0,
        },
    )
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 9.0, "topic_route", "official")],
        retrieved_chunks=[chunk],
    )
    draft = LLMAnswerDraft(
        direct_answer="O prazo de recurso é de 10 dias.",
        simple_explanation="Há um prazo de 10 dias para recorrer da prisão preventiva.",
    )

    validation = legal_validation_service.validate(classification, retrieval, draft)

    assert any(issue.code == "processual_specificity_gap" for issue in validation.issues)
    assert validation.answer_mode == "limited"


async def test_cpp_recurso_question_without_recurso_in_same_context_is_limited():
    classification = await legal_classifier.classify("Qual é o prazo de recurso no CPP para prisão preventiva?")
    chunk = RetrievedChunk(
        chunk_id="cpp-283-only",
        text="ARTIGO 283.º (Prazos máximos de prisão preventiva) A prisão preventiva cessa quando decorrerem 4, 6, 12 ou 18 meses, conforme a fase do processo.",
        source="Codigo-Processo-Penal.pdf",
        title="Codigo do Processo Penal (Lei 39/20)",
        link_original="https://example.test/cpp.pdf",
        page=122,
        article_number="283",
        law_status="Vigente",
        distance=0.1,
        source_scope="official",
        document_id=None,
        metadata={
            "legal_branch": "penal",
            "diploma_slug": "codigo-processo-penal-lei-39-20",
            "segmentation": "article_block",
            "article_main": "283",
            "article_references": ["283"],
            "is_normative": True,
            "normative_density": 4.0,
        },
    )
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 9.0, "topic_route", "official")],
        retrieved_chunks=[chunk],
    )
    draft = LLMAnswerDraft(
        direct_answer="O prazo de recurso é de 4 meses sem acusação e 6 meses sem pronúncia.",
        simple_explanation="Os prazos da prisão preventiva variam conforme a fase do processo.",
        cited_articles=["283"],
    )

    validation = legal_validation_service.validate(classification, retrieval, draft)

    assert any(issue.code == "processual_specificity_gap" for issue in validation.issues)
    assert validation.answer_mode == "limited"


async def test_cpp_phrase_prazo_de_recurso_requires_exact_support():
    classification = await legal_classifier.classify("Qual é o prazo de recurso no CPP para prisão preventiva?")
    chunk = RetrievedChunk(
        chunk_id="cpp-near-miss",
        text="ARTIGO 268.º (Extinção das medidas de coacção) A sentença absolutória, mesmo havendo recurso, extingue medidas de prisão preventiva. Os prazos seguem o regime legal aplicável.",
        source="Codigo-Processo-Penal.pdf",
        title="Codigo do Processo Penal (Lei 39/20)",
        link_original="https://example.test/cpp.pdf",
        page=120,
        article_number="268",
        law_status="Vigente",
        distance=0.1,
        source_scope="official",
        document_id=None,
        metadata={
            "legal_branch": "penal",
            "diploma_slug": "codigo-processo-penal-lei-39-20",
            "segmentation": "article_block",
            "article_main": "268",
            "article_references": ["268"],
            "is_normative": True,
            "normative_density": 4.0,
        },
    )
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 9.0, "topic_route", "official")],
        retrieved_chunks=[chunk],
    )
    draft = LLMAnswerDraft(
        direct_answer="O prazo de recurso é de 30 dias.",
        simple_explanation="Há recurso e há prazos processuais, mas o contexto nao mostra o prazo de recurso.",
    )

    validation = legal_validation_service.validate(classification, retrieval, draft)

    assert any(issue.code == "processual_specificity_gap" for issue in validation.issues)
    assert validation.answer_mode == "limited"


async def test_answer_tracking_detects_off_topic_draft():
    classification = await legal_classifier.classify("Se a resposta depender de lei processual e lei substantiva de ramos diferentes, como o sistema lida com mistura de fontes?")
    draft = LLMAnswerDraft(
        direct_answer="Sim, um mútuo sem contrato escrito pode ser exigido judicialmente em Angola.",
        simple_explanation="Transferências bancárias e mensagens podem servir como prova.",
    )

    assert legal_composer.answer_tracks_question(
        "Se a resposta depender de lei processual e lei substantiva de ramos diferentes, como o sistema lida com mistura de fontes?",
        draft,
        classification,
    ) is False


async def test_composer_discards_articles_outside_retrieved_context():
    chunk = _chunk("Lei Geral do Trabalho (Lei 12/23)", "Artigo 20 O empregador ...", "20", "official", "laboral")
    draft = LLMAnswerDraft(cited_articles=["99", "20"], cited_diplomas=["Lei Geral do Trabalho", "Código Penal"])

    constrained = legal_composer.constrain_draft_to_context(draft, [chunk])

    assert constrained.cited_articles == ["20"]
    assert constrained.cited_diplomas == ["Lei Geral do Trabalho"]


async def test_validation_no_longer_flags_filtered_articles():
    classification = await legal_classifier.classify("Qual o artigo aplicável ao despedimento sem aviso?")
    chunk = _chunk("Lei Geral do Trabalho (Lei 12/23)", "Artigo 12 O trabalhador tem direito ...", "12", "official", "laboral")
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 8.0, "base", "official")],
        retrieved_chunks=[chunk],
    )
    draft = legal_composer.constrain_draft_to_context(
        LLMAnswerDraft(direct_answer="Resposta curta", cited_articles=["99"], cited_diplomas=["Lei Geral do Trabalho"]),
        retrieval.retrieved_chunks,
    )

    validation = legal_validation_service.validate(classification, retrieval, draft)

    assert validation.unsupported_articles == []
    assert not any(issue.code == "unsupported_article" for issue in validation.issues)


# Retain explicit validator coverage for unsupported raw LLM citations.
async def test_flags_unsupported_articles():
    classification = await legal_classifier.classify("Qual o artigo aplicável ao despedimento sem aviso?")
    chunk = _chunk("Lei Geral do Trabalho (Lei 12/23)", "Artigo 12 O trabalhador tem direito ...", "12")
    retrieval = RetrievalResult(
        classification=classification,
        official_evidence=[RetrievalEvidence("q", chunk, 8.0, "base", "official")],
        retrieved_chunks=[chunk],
    )
    draft = LLMAnswerDraft(
        direct_answer="Resposta curta",
        cited_articles=["99"],
        cited_diplomas=["Lei Geral do Trabalho"],
    )

    validation = legal_validation_service.validate(classification, retrieval, draft)

    assert validation.unsupported_articles == ["99"]
    assert any(issue.code == "unsupported_article" for issue in validation.issues)
