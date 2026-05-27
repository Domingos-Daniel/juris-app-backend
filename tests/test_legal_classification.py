import pytest
pytestmark = pytest.mark.asyncio
from app.services.legal.classification import legal_classifier


async def test_classifies_labor_question_for_layperson():
    result = await legal_classifier.classify("Fui despedido sem aviso e sem indemnização. O que faço agora?")

    assert result.main_branch == "laboral"
    assert result.request_type == "passos_praticos"
    assert result.audience == "leigo"
    assert result.needs_practical_guidance is True


async def test_classifies_multi_branch_comparison():
    result = await legal_classifier.classify("Qual a diferença entre responsabilidade penal e laboral no não pagamento de indemnização?")

    assert result.main_branch == "misto"
    assert set(result.branch_candidates) >= {"penal", "laboral"}
    assert result.specificity == "comparacao_multi_ramo"
    assert result.request_type == "comparacao"


async def test_classifies_technical_legal_basis_validation():
    result = await legal_classifier.classify("Confirme a base legal e o artigo aplicável ao mútuo sem contrato escrito no Código Civil.")

    assert result.main_branch == "civil"
    assert result.specificity == "validacao_base_legal"
    assert result.audience == "tecnico"
    assert result.needs_article_validation is True


async def test_short_query_with_own_domain_signal_is_not_forced_to_history_follow_up():
    history = ["Utilizador: Fui despedido sem aviso e sem indemnização."]
    result = await legal_classifier.classify("Um mútuo sem contrato escrito pode ser exigido judicialmente?", history)

    assert result.main_branch == "civil"
    assert result.specificity != "follow_up"
    assert result.conversation_branch_hint is None


async def test_domain_comparison_question_is_not_misclassified_as_follow_up():
    result = await legal_classifier.classify("Qual a diferença entre compensação, indemnização e reintegração no ramo laboral angolano?")

    assert result.main_branch == "laboral"
    assert result.specificity == "geral"
    assert result.request_type == "comparacao"


async def test_real_follow_up_can_use_history_branch_hint():
    history = ["Utilizador: Fui despedido sem aviso e sem indemnização."]
    result = await legal_classifier.classify("Explique melhor.", history)

    assert result.specificity == "follow_up"
    assert result.conversation_branch_hint == "laboral"


async def test_referential_follow_up_inherits_anchor_branch_and_request_type():
    history = [
        "Utilizador: No mesmo caso, o que é laboral e o que é penal quando há despedimento e falta de pagamento?",
        "Assistente: resposta",
    ]
    result = await legal_classifier.classify("Qual a diferença na prática?", history)

    assert result.specificity == "follow_up"
    assert result.main_branch == "misto"
    assert set(result.branch_candidates) >= {"laboral", "penal"}
    assert result.request_type == "explicacao_simples"
    assert result.conversation_branch_hint == "misto"


async def test_referential_follow_up_keeps_practical_guidance_when_anchor_was_practical():
    history = [
        "Utilizador: Fui despedido sem aviso e sem indemnização. O que faço agora?",
        "Assistente: resposta",
    ]
    result = await legal_classifier.classify("Explique melhor.", history)

    assert result.specificity == "follow_up"
    assert result.main_branch == "laboral"
    assert result.needs_practical_guidance is True
    assert result.request_type == "passos_praticos"


async def test_cpp_question_sets_strict_process_route():
    result = await legal_classifier.classify("Qual é o prazo de recurso no CPP para prisão preventiva?")

    assert result.main_branch == "penal"
    assert result.topic_route == "cpp"
    assert result.norm_type_needed == "processual"
    assert result.requires_strict_corpus_match is True
    assert "Código do Processo Penal" in result.requested_diplomas


async def test_bi_question_routes_to_identificacao_civil():
    result = await legal_classifier.classify("Qual é o custo da segunda via do bilhete de identidade em Angola?")

    assert result.main_branch == "administrativo"
    assert result.topic_route == "identificacao_civil"
    assert result.norm_type_needed == "administrativo_operacional"
    assert result.requires_strict_corpus_match is True
    assert "Lei do Bilhete de Identidade" in result.requested_diplomas


async def test_heranca_question_routes_to_sucessoes_with_strict_family_corpus():
    result = await legal_classifier.classify("Como funciona herança em Angola?")

    assert result.main_branch == "familia"
    assert result.topic_route == "sucessoes"
    assert result.specificity == "sucessorio"
    assert result.requires_strict_corpus_match is True
    assert "Código de Família" in result.requested_diplomas


async def test_drafting_request_enables_drafting_mode_without_strict_corpus():
    result = await legal_classifier.classify("Preciso de uma minuta de procuração para representar um familiar.")

    assert result.request_type == "minuta_documental"
    assert result.topic_route == "drafting"
    assert result.drafting_mode is True
    assert result.requires_strict_corpus_match is False


async def test_follow_up_inherits_topic_route_and_norm_type_from_anchor():
    history = [
        "Utilizador: Qual é o custo da segunda via do bilhete de identidade em Angola?",
        "Assistente: resposta",
    ]
    result = await legal_classifier.classify("Explique melhor.", history)

    assert result.specificity == "follow_up"
    assert result.main_branch == "administrativo"
    assert result.topic_route == "identificacao_civil"
    assert result.norm_type_needed == "administrativo_operacional"
    assert result.conversation_topic_hint == "identificacao_civil"
    assert result.conversation_norm_type_hint == "administrativo_operacional"
    assert "Lei do Bilhete de Identidade" in result.requested_diplomas


async def test_constitutional_question_does_not_false_match_iva_inside_other_words():
    result = await legal_classifier.classify(
        "Se uma pessoa é privada da liberdade por autoridade pública em condições ilegais, quais garantias constitucionais e tutela jurisdicional efetiva podem ser acionadas?"
    )

    assert result.main_branch == "constitucional"
    assert result.topic_route == "constitucional"
    assert "Constituição da República de Angola" in result.requested_diplomas
    assert "Código do IVA" not in result.requested_diplomas


async def test_tax_question_does_not_get_diverted_to_societies_by_generic_empresa_marker():
    result = await legal_classifier.classify(
        "Uma pequena empresa angolana quer saber quais obrigações fiscais mínimas e deveres declarativos podem surgir em matéria tributária e de IVA."
    )

    assert result.main_branch == "tributario"
    assert result.topic_route in {"tributario", "iva"}
    assert "Lei das Sociedades Comerciais" not in result.requested_diplomas


async def test_meta_system_question_is_classified_as_meta_sistema():
    result = await legal_classifier.classify(
        "Se a resposta depender de lei processual e lei substantiva de ramos diferentes, como o sistema lida com mistura de fontes e que alertas devolve?"
    )

    assert result.specificity == "meta_sistema"
    assert result.requires_strict_corpus_match is False


async def test_burla_question_routes_to_penal_substantivo():
    result = await legal_classifier.classify("Qual é a pena para burla?")

    assert result.main_branch == "penal"
    assert result.topic_route == "penal_substantivo"
    assert "Código Penal" in result.requested_diplomas
