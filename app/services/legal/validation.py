from __future__ import annotations

import re

from app.services.legal.models import (
    LegalClassification,
    LLMAnswerDraft,
    RetrievalResult,
    ValidatedLegalBasisItem,
    ValidationIssue,
    ValidationResult,
)
from app.services.legal.normative_guardrails import normative_guardrails_service

ARTICLE_RE = re.compile(r"(?:art|artigo|artigos)\s*(\d+[.]?\d*)", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+[.,]?\d*\b")

# Branches that are sub-domains of parent branches.
# When the classifier says "civil" and the chunk says "propriedade", they are compatible.
_BRANCH_PARENTS: dict[str, str] = {
    "propriedade": "civil",
    "sucessorio": "civil",
}


def _branches_compatible(a: str, b: str) -> bool:
    """Check if two branches are compatible (one is parent/child of the other)."""
    return _BRANCH_PARENTS.get(a) == b or _BRANCH_PARENTS.get(b) == a


CPP_STRICT_TERMS = (
    "prisão preventiva",
    "prisao preventiva",
    "medidas de coacção",
    "medidas de coaccao",
    "medidas de coação",
    "medidas de coacao",
    "recurso",
    "prazo",
)
CPP_CORE_TERMS = (
    "prisão preventiva",
    "prisao preventiva",
    "medidas de coacção",
    "medidas de coaccao",
    "medidas de coação",
    "medidas de coacao",
)
CPP_RESOURCE_TERMS = ("recurso", "prazo")
PENAL_MATERIAL_SLUG = "codigo-penal-lei-38-20"
PENAL_MATERIAL_ARTICLES = {"417", "418", "426", "406", "399", "468"}
PENAL_MATERIAL_TERMS = (
    "burla",
    "infidelidade",
    "apropriação",
    "apropriacao",
    "retenção",
    "retencao",
    "vantagem patrimonial",
    "prejuízo patrimonial",
    "prejuizo patrimonial",
    "enriquecimento ilícito",
    "enriquecimento ilicito",
)
PENAL_DIRECT_QUERY_TERMS = {
    "burla": {"417", "418"},
    "infidelidade": {"426"},
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_articles(text: str) -> set[str]:
    return {
        match.group(1).replace(".", "") for match in ARTICLE_RE.finditer(text or "")
    }


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = item.strip().replace(".", "")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _chunk_branch(chunk) -> str:
    metadata = chunk.metadata or {}
    branch = metadata.get("legal_branch")
    if branch:
        return branch
    haystack = _normalize(f"{chunk.title} {chunk.source}")
    if "trabalho" in haystack:
        return "laboral"
    if "penal" in haystack:
        return "penal"
    if "civil" in haystack:
        return "civil"
    if "constitu" in haystack:
        return "constitucional"
    return "indeterminado"


def _article_references(chunk) -> list[str]:
    metadata = chunk.metadata or {}
    refs = metadata.get("article_references") or []
    if refs:
        return _ordered_unique([str(item) for item in refs])
    combined: list[str] = []
    if chunk.article_number:
        combined.extend(
            part.strip() for part in chunk.article_number.split(",") if part.strip()
        )
    combined.extend(sorted(_extract_articles(chunk.text)))
    return _ordered_unique(combined)


def _primary_article(chunk) -> str | None:
    metadata = chunk.metadata or {}
    article_main = metadata.get("article_main")
    if article_main:
        return str(article_main).replace(".", "")
    refs = _article_references(chunk)
    return refs[0] if refs else None


def _has_primary_anchor(chunk) -> bool:
    primary = _primary_article(chunk)
    if not primary:
        return False
    text = _normalize(chunk.text)
    return any(
        text.startswith(pattern)
        for pattern in (f"artigo {primary}", f"art. {primary}", f"art {primary}")
    )


def _is_promulgation_or_cross_reference_chunk(chunk) -> bool:
    if (chunk.metadata or {}).get("document_kind") == "jurisprudence":
        return True
    text = _normalize(chunk.text)
    refs = _article_references(chunk)
    if (chunk.metadata or {}).get("is_front_matter") or (chunk.metadata or {}).get(
        "is_structural"
    ):
        return True
    if (chunk.page or 0) <= 3 and len(refs) >= 2:
        return True
    return any(
        token in text
        for token in (
            "aprova o código",
            "aprova o codigo",
            "é revogado",
            "e revogado",
            "diário da república",
            "diario da republica",
            "assembleia nacional",
            "publicação",
            "publicacao",
        )
    )


def _is_strictly_confirmable(chunk) -> bool:
    if _is_promulgation_or_cross_reference_chunk(chunk):
        return False
    metadata = chunk.metadata or {}
    segmentation = metadata.get("segmentation")
    refs = _article_references(chunk)
    if (
        segmentation == "article_block"
        and _has_primary_anchor(chunk)
        and len(refs) <= 4
    ):
        return True
    if _has_primary_anchor(chunk) and len(refs) <= 5:
        return True
    return len(refs) <= 2


def _is_prudentially_confirmable(chunk) -> bool:
    if _is_promulgation_or_cross_reference_chunk(chunk):
        return False
    return bool(_primary_article(chunk) or _article_references(chunk))


def _is_operational_administrative_chunk(
    classification: LegalClassification, chunk
) -> bool:
    if classification.topic_route not in {
        "identificacao_civil",
        "contencioso_admin",
        "processo_administrativo",
    }:
        return True
    if _is_promulgation_or_cross_reference_chunk(chunk):
        return False
    text = _normalize(chunk.text)
    refs = _article_references(chunk)
    if classification.topic_route == "identificacao_civil":
        return bool(refs) and any(
            term in text
            for term in (
                "bilhete de identidade",
                "segunda via",
                "emissão",
                "emissao",
                "taxa",
                "custo",
                "pagamento",
                "requerente",
            )
        )
    if classification.topic_route == "contencioso_admin":
        return bool(refs) and any(
            term in text
            for term in (
                "acto administrativo",
                "ato administrativo",
                "impugna",
                "recurso",
                "prazo",
                "notificação",
                "notificacao",
            )
        )
    return bool(refs) and any(
        term in text
        for term in (
            "procedimento administrativo",
            "órgão",
            "orgao",
            "licença",
            "licenca",
            "prazo",
            "notificação",
            "notificacao",
        )
    )


def _build_deep_link_for_chunk(chunk) -> str | None:
    if chunk.link_original and chunk.page and "#page=" not in chunk.link_original:
        return f"{chunk.link_original}#page={chunk.page}"
    return chunk.link_original


def _build_basis_item(
    chunk, confirmed: bool, article: str | None
) -> ValidatedLegalBasisItem:
    return ValidatedLegalBasisItem(
        diploma=chunk.title,
        article=article,
        page=chunk.page,
        source_scope=chunk.source_scope,
        confirmed=confirmed,
        excerpt=chunk.text[:420],
        deep_link=_build_deep_link_for_chunk(chunk),
    )


def _effective_unsupported_articles(
    cited_articles: list[str], context_articles: set[str], retrieval: RetrievalResult
) -> list[str]:
    unsupported: list[str] = []
    for article in cited_articles:
        if article in context_articles:
            continue
        if any(
            article in _article_references(evidence.chunk)
            for evidence in retrieval.official_evidence
        ):
            continue
        unsupported.append(article)
    return unsupported


def _general_knowledge_issue(validation_needed: bool) -> ValidationIssue:
    return ValidationIssue(
        code="general_knowledge_boundary",
        message="A resposta pode usar enquadramento jurídico geral do modelo, mas a confirmação normativa específica exige validação adicional no contexto recuperado.",
        severity="medium" if validation_needed else "low",
    )


def _limited_context_issue() -> ValidationIssue:
    return ValidationIssue(
        code="limited_contextual_support",
        message="Há enquadramento jurídico plausível, mas o contexto recuperado não basta para confirmar integralmente a formulação normativa usada na resposta.",
        severity="medium",
    )


def _multi_article_issue() -> ValidationIssue:
    return ValidationIssue(
        code="multi_article_chunk",
        message="Parte da base legal recuperada junta vários artigos no mesmo excerto; isso reduz a precisão da confirmação normativa específica.",
        severity="medium",
    )


def _weak_confirmation_issue() -> ValidationIssue:
    return ValidationIssue(
        code="weak_article_confirmation",
        message="Há base legal contextual relevante, mas a confirmação estrita do artigo principal ainda é insuficiente.",
        severity="medium",
    )


def _strict_gap_issue() -> ValidationIssue:
    return ValidationIssue(
        code="strict_confirmation_gap",
        message="A resposta continua útil, mas a confirmação normativa específica não está suficientemente isolada em excertos de artigo único.",
        severity="medium",
    )


def _strict_corpus_mismatch_issue() -> ValidationIssue:
    return ValidationIssue(
        code="strict_corpus_mismatch",
        message="A rota exige correspondência estrita com diploma prioritário, mas a evidência oficial recuperada não corresponde ao diploma pedido.",
        severity="high",
    )


def _strict_corpus_missing_issue() -> ValidationIssue:
    return ValidationIssue(
        code="strict_corpus_missing",
        message="A rota exige correspondência estrita com diploma prioritário, mas não foi recuperada evidência oficial compatível suficiente.",
        severity="high",
    )


def _requested_diploma_slugs(classification: LegalClassification) -> set[str]:
    slugs: set[str] = set()
    diploma_map = {
        "Lei Geral do Trabalho": "lei-geral-do-trabalho-lei-12-23",
        "Código Penal": "codigo-penal-lei-38-20",
        "Codigo Penal": "codigo-penal-lei-38-20",
        "Código Civil": "codigo-civil",
        "Codigo Civil": "codigo-civil",
        "Constituição da República de Angola": "constituicao-republica-angola-2022",
        "Constituicao da Republica de Angola": "constituicao-republica-angola-2022",
        "Código do Processo Penal": "codigo-processo-penal-lei-39-20",
        "Codigo do Processo Penal": "codigo-processo-penal-lei-39-20",
        "Código de Processo Civil": "codigo-processo-civil",
        "Codigo de Processo Civil": "codigo-processo-civil",
        "Lei do Contencioso Administrativo": "codigo-processo-contencioso-administrativo-33-22",
        "Lei do Processo Administrativo": "lei-processo-administrativo-lei-2-22",
        "Lei do Bilhete de Identidade": "lei-bilhete-identidade-4-16",
        "Estatuto dos Magistrados Judiciais": "estatuto-magistrados-judiciais-lei-7-94",
        "Lei Orgânica do Tribunal Supremo": "lei-organica-tribunal-supremo",
        "Lei Organica do Tribunal Supremo": "lei-organica-tribunal-supremo",
        "Lei das Sociedades Comerciais": "lei-sociedades-comerciais-1-04",
        "Código Geral Tributário": "codigo-geral-tributario-21-14",
        "Codigo Geral Tributario": "codigo-geral-tributario-21-14",
        "Código do IVA": "codigo-iva-lei-7-19",
        "Codigo do IVA": "codigo-iva-lei-7-19",
        "Código de Família": "codigo-familia-lei-1-88",
        "Codigo de Familia": "codigo-familia-lei-1-88",
        "Lei de Terras": "lei-terras-9-04",
    }
    for diploma in classification.requested_diplomas:
        slug = diploma_map.get(diploma)
        if slug:
            slugs.add(slug)
    return slugs


def _strict_corpus_matches(
    classification: LegalClassification, retrieval: RetrievalResult
) -> bool:
    requested = _requested_diploma_slugs(classification)
    if not requested:
        return bool(retrieval.official_evidence)
    return any(
        (evidence.chunk.metadata or {}).get("diploma_slug") in requested
        for evidence in retrieval.official_evidence
    )


def _strict_corpus_dominant_mismatch(
    classification: LegalClassification, retrieval: RetrievalResult
) -> bool:
    if not retrieval.official_evidence:
        return False
    requested = _requested_diploma_slugs(classification)
    if not requested:
        return False
    dominant_slug = (retrieval.official_evidence[0].chunk.metadata or {}).get(
        "diploma_slug"
    )
    return bool(dominant_slug and dominant_slug not in requested)


def _penal_gap_issue() -> ValidationIssue:
    return ValidationIssue(
        code="penal_relevance_gap",
        message="A cobertura penal recuperada ainda não isola um fundamento penal material suficientemente forte para alta confiança.",
        severity="medium",
    )


def _cpp_specificity_issue() -> ValidationIssue:
    return ValidationIssue(
        code="processual_specificity_gap",
        message="A base processual recuperada ainda não confirma com precisão suficiente o ponto específico perguntado sobre prisão preventiva no CPP.",
        severity="medium",
    )


def _issue_once(issues: list[ValidationIssue], issue: ValidationIssue) -> None:
    if any(existing.code == issue.code for existing in issues):
        return
    issues.append(issue)


def _dedupe_basis(
    items: list[ValidatedLegalBasisItem],
) -> list[ValidatedLegalBasisItem]:
    deduped: list[ValidatedLegalBasisItem] = []
    seen: set[tuple[str, str | None, int | None, bool]] = set()
    for item in items:
        key = (item.diploma, item.article, item.page, item.confirmed)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:5]


def _limit_basis_for_route(
    classification: LegalClassification, items: list[ValidatedLegalBasisItem]
) -> list[ValidatedLegalBasisItem]:
    if classification.topic_route == "penal_substantivo":
        required_articles = _required_penal_articles_for_query(classification)
        if required_articles:
            preferred = [
                item
                for item in items
                if item.article
                and str(item.article).replace(".", "") in required_articles
            ]
            if preferred:
                return preferred[:3]
        return items[:3]
    if classification.topic_route == "cpp":
        return items[:3]
    return items


def _compress_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    priority = {
        "unsupported_article": 0,
        "followup_anchor_unresolved": 1,
        "normative_conflict": 2,
        "citator_gap": 3,
        "vigency_unverified": 4,
        "branch_gap": 1,
        "mixed_sources": 2,
        "branch_mismatch": 3,
        "strict_corpus_missing": 4,
        "strict_corpus_mismatch": 5,
        "penal_relevance_gap": 6,
        "strict_confirmation_gap": 7,
        "weak_article_confirmation": 8,
        "multi_article_chunk": 9,
        "general_knowledge_boundary": 10,
        "limited_contextual_support": 11,
        "no_official_support": 12,
    }
    return sorted(issues, key=lambda item: priority.get(item.code, 99))[:5]


def _answer_mode(
    classification: LegalClassification,
    retrieval: RetrievalResult,
    issues: list[ValidationIssue],
    confirmed: list[ValidatedLegalBasisItem],
    sufficient_legal_support: bool,
) -> str:
    issue_codes = {issue.code for issue in issues}
    high_issue_codes = {
        "unsupported_article",
        "no_official_support",
        "followup_anchor_unresolved",
        "normative_conflict",
        "citator_gap",
        "vigency_unverified",
    }
    has_high_blocker = any(code in issue_codes for code in high_issue_codes)

    if not retrieval.official_evidence:
        return "limited"
    if has_high_blocker:
        return "grounded_with_caveat"
    if classification.requires_strict_corpus_match and not confirmed:
        return "limited"
    if classification.needs_article_validation and not confirmed:
        return "limited"
    if classification.needs_multi_branch_handling and retrieval.missing_branches:
        return "limited"
    if classification.specificity == "meta_sistema":
        return "limited"
    if (
        "penal_relevance_gap" in issue_codes
        or "processual_specificity_gap" in issue_codes
        or "branch_mismatch" in issue_codes
    ):
        return "grounded_with_caveat"
    if sufficient_legal_support and confirmed:
        return "grounded"
    return "limited"


def _is_material_penal_chunk(chunk) -> bool:
    if _chunk_branch(chunk) != "penal":
        return True
    text = _normalize(chunk.text)
    refs = _article_references(chunk)
    metadata = chunk.metadata or {}
    if (chunk.metadata or {}).get("is_front_matter"):
        return False
    if any(
        token in text
        for token in (
            "revogação da legislação",
            "revogacao da legislacao",
            "disposições legais",
            "disposicoes legais",
        )
    ):
        return False
    if metadata.get("diploma_slug") != PENAL_MATERIAL_SLUG:
        return False
    if not refs or len(refs) > 3:
        return False
    if any(ref in PENAL_MATERIAL_ARTICLES for ref in refs):
        return True
    return any(token in text for token in PENAL_MATERIAL_TERMS)


def _question_requires_cpp_specificity(
    classification: LegalClassification, answer_draft: LLMAnswerDraft
) -> bool:
    if classification.topic_route != "cpp":
        return False
    query_haystack = _normalize(classification.query_text)
    if any(term in query_haystack for term in CPP_RESOURCE_TERMS):
        return True
    haystack = _normalize(answer_draft.rich_content)
    return any(term in haystack for term in CPP_RESOURCE_TERMS)


def _required_cpp_query_terms(answer_draft: LLMAnswerDraft) -> tuple[str, ...]:
    haystack = _normalize(answer_draft.rich_content)
    required: list[str] = []
    if "recurso" in haystack or "recorrer" in haystack:
        required.append("recurso")
    if "prazo" in haystack:
        required.append("prazo")
    return tuple(required)


def _required_cpp_terms_for_classification(
    classification: LegalClassification, answer_draft: LLMAnswerDraft
) -> tuple[str, ...]:
    query_haystack = _normalize(classification.query_text)
    required: list[str] = []
    if "recurso" in query_haystack or "recorrer" in query_haystack:
        required.append("recurso")
    if "prazo" in query_haystack:
        required.append("prazo")
    if required:
        return tuple(required)
    return _required_cpp_query_terms(answer_draft)


def _has_cpp_specific_support(
    classification: LegalClassification,
    retrieval: RetrievalResult,
    answer_draft: LLMAnswerDraft,
) -> bool:
    required_terms = _required_cpp_terms_for_classification(
        classification, answer_draft
    )
    for evidence in retrieval.official_evidence:
        chunk = evidence.chunk
        if _chunk_branch(chunk) != "penal":
            continue
        text = _normalize(chunk.text)
        if any(
            term in text
            for term in (
                "revogação da legislação",
                "revogacao da legislacao",
                "substituição do perito",
                "substituicao do perito",
            )
        ):
            continue
        if not all(term in text for term in required_terms):
            continue
        if any(term in text for term in CPP_CORE_TERMS) and any(
            term in text for term in CPP_RESOURCE_TERMS
        ):
            return True
    return False


def _confirmed_basis_matches_cpp_query(
    classification: LegalClassification,
    confirmed: list[ValidatedLegalBasisItem],
    retrieval: RetrievalResult,
) -> bool:
    if classification.topic_route != "cpp":
        return True
    required_terms = _required_cpp_terms_for_classification(
        classification, LLMAnswerDraft()
    )
    if not required_terms:
        return True
    index: dict[tuple[str, int | None, str], str] = {}
    for evidence in retrieval.official_evidence:
        chunk = evidence.chunk
        index[(chunk.title, chunk.page, chunk.source_scope)] = _normalize(chunk.text)
    for item in confirmed:
        text = index.get((item.diploma, item.page, item.source_scope), "")
        if text and all(term in text for term in required_terms):
            return True
    return False


def _cpp_query_semantically_supported(
    classification: LegalClassification, retrieval: RetrievalResult
) -> bool:
    if classification.topic_route != "cpp":
        return True
    query = _normalize(classification.query_text)
    needs_recurso = "recurso" in query
    needs_prazo = "prazo" in query
    if not (needs_recurso or needs_prazo):
        return True
    for evidence in retrieval.official_evidence:
        text = _normalize(evidence.chunk.text)
        if needs_recurso and "recurso" not in text:
            continue
        if needs_prazo and "prazo" not in text and "prazos" not in text:
            continue
        if any(
            term in text
            for term in (
                "prisão preventiva",
                "prisao preventiva",
                "medidas de coacção",
                "medidas de coaccao",
                "medidas de coação",
                "medidas de coacao",
            )
        ):
            return True
    return False


def _cpp_exact_phrase_supported(
    classification: LegalClassification,
    retrieval: RetrievalResult,
    answer_draft: LLMAnswerDraft,
) -> bool:
    if classification.topic_route != "cpp":
        return True
    query = _normalize(classification.query_text)
    answer = _normalize(answer_draft.rich_content)
    if "prazo de recurso" not in query and "prazo de recurso" not in answer:
        return True
    for evidence in retrieval.official_evidence:
        text = _normalize(evidence.chunk.text)
        if "prazo de recurso" in text:
            return True
    return False


def _numeric_claims_supported_in_cpp_context(
    classification: LegalClassification,
    retrieval: RetrievalResult,
    answer_draft: LLMAnswerDraft,
) -> bool:
    if classification.topic_route != "cpp":
        return True
    query_haystack = _normalize(classification.query_text)
    if not any(term in query_haystack for term in CPP_RESOURCE_TERMS):
        return True
    answer_text = _normalize(answer_draft.rich_content)
    answer_numbers = {
        match.group(0).replace(",", ".") for match in NUMBER_RE.finditer(answer_text)
    }
    if not answer_numbers:
        return True
    context_text = "\n".join(
        _normalize(evidence.chunk.text) for evidence in retrieval.official_evidence
    )
    return any(number in context_text for number in answer_numbers)


def _required_penal_articles_for_query(classification: LegalClassification) -> set[str]:
    if classification.topic_route != "penal_substantivo":
        return set()
    haystack = _normalize(classification.query_text)
    required: set[str] = set()
    for term, articles in PENAL_DIRECT_QUERY_TERMS.items():
        if term in haystack:
            required.update(articles)
    return required


def _penal_substantivo_matches_query(
    classification: LegalClassification, confirmed: list[ValidatedLegalBasisItem]
) -> bool:
    required_articles = _required_penal_articles_for_query(classification)
    if not required_articles:
        return True
    confirmed_articles = {
        str(item.article).replace(".", "") for item in confirmed if item.article
    }
    return bool(confirmed_articles.intersection(required_articles))


class LegalValidationService:
    def validate(
        self,
        classification: LegalClassification,
        retrieval: RetrievalResult,
        answer_draft: LLMAnswerDraft,
    ) -> ValidationResult:
        confirmed: list[ValidatedLegalBasisItem] = []
        prudential: list[ValidatedLegalBasisItem] = []
        issues: list[ValidationIssue] = []

        # Se for uma correcção, marcamos para leniência em certos checks semânticos
        is_correction = classification.is_correction

        official_sources = 0
        user_sources = 0
        context_articles: set[str] = set()

        for evidence in retrieval.official_evidence:
            chunk = evidence.chunk
            official_sources += 1
            context_articles.update(_extract_articles(chunk.text))
            context_articles.update(_article_references(chunk))

        for evidence in retrieval.user_evidence:
            user_sources += 1

        cited_articles = [
            article.replace(".", "")
            for article in answer_draft.cited_articles
            if article
        ]
        unsupported_articles = _effective_unsupported_articles(
            cited_articles, context_articles, retrieval
        )
        if unsupported_articles:
            issues.append(
                ValidationIssue(
                    code="unsupported_article",
                    message="Há artigos citados sem suporte verificável no contexto recuperado.",
                    severity="high",
                )
            )

        if classification.needs_multi_branch_handling and retrieval.missing_branches:
            issues.append(
                ValidationIssue(
                    code="branch_gap",
                    message="Há lacunas de contexto em pelo menos um dos ramos pedidos.",
                    severity="medium",
                )
            )

        source_cross_contamination = bool(official_sources and user_sources)
        if source_cross_contamination and classification.needs_source_separation:
            issues.append(
                ValidationIssue(
                    code="mixed_sources",
                    message="Foram usadas simultaneamente fontes oficiais e documentos do utilizador; a composição final deve distingui-las claramente.",
                    severity="medium",
                )
            )

        dominant_official_branch = (
            _chunk_branch(retrieval.official_evidence[0].chunk)
            if retrieval.official_evidence
            else None
        )
        if (
            classification.main_branch not in {"misto", "indeterminado"}
            and dominant_official_branch
            and dominant_official_branch != classification.main_branch
            and not _branches_compatible(
                classification.main_branch, dominant_official_branch
            )
        ):
            issues.append(
                ValidationIssue(
                    code="branch_mismatch",
                    message="A fonte oficial dominante não coincide com o ramo jurídico principal classificado.",
                    severity="high",
                )
            )

        (
            normative_status,
            normative_notes,
            normative_issues,
            jurisprudence_basis,
        ) = normative_guardrails_service.analyze(classification, retrieval, confirmed)
        for issue in normative_issues:
            _issue_once(issues, issue)

        if classification.requires_strict_corpus_match:
            if not _strict_corpus_matches(classification, retrieval):
                _issue_once(issues, _strict_corpus_missing_issue())
            elif _strict_corpus_dominant_mismatch(classification, retrieval):
                _issue_once(issues, _strict_corpus_mismatch_issue())

        multi_article_noise_detected = False
        for evidence in retrieval.official_evidence[:5]:
            chunk = evidence.chunk
            if not _is_operational_administrative_chunk(classification, chunk):
                continue
            article = _primary_article(chunk)
            strict_confirmed = _is_strictly_confirmable(chunk) and bool(article)
            if strict_confirmed:
                confirmed.append(_build_basis_item(chunk, True, article))
            elif _is_prudentially_confirmable(chunk):
                prudential.append(_build_basis_item(chunk, False, article))
            if len(_article_references(chunk)) >= 3 and not _has_primary_anchor(chunk):
                multi_article_noise_detected = True

        if (
            not confirmed
            and not classification.needs_article_validation
            and retrieval.official_evidence
            and prudential
        ):
            promoted = []
            for item in prudential[:2]:
                promoted.append(item.model_copy(update={"confirmed": True}))
            confirmed.extend(promoted)

        penal_gap = (
            classification.main_branch == "misto"
            and any(branch == "penal" for branch in classification.branch_candidates)
            and not any(
                _is_material_penal_chunk(evidence.chunk)
                for evidence in retrieval.official_evidence
                if _chunk_branch(evidence.chunk) == "penal"
            )
        )
        if penal_gap:
            _issue_once(issues, _penal_gap_issue())

        if _question_requires_cpp_specificity(
            classification, answer_draft
        ) and not _has_cpp_specific_support(classification, retrieval, answer_draft):
            _issue_once(issues, _cpp_specificity_issue())
        elif (
            classification.topic_route == "cpp"
            and not _confirmed_basis_matches_cpp_query(
                classification, confirmed, retrieval
            )
        ):
            _issue_once(issues, _cpp_specificity_issue())
        if (
            classification.topic_route == "cpp"
            and not _cpp_query_semantically_supported(classification, retrieval)
        ):
            _issue_once(issues, _cpp_specificity_issue())
        if classification.topic_route == "cpp" and not _cpp_exact_phrase_supported(
            classification, retrieval, answer_draft
        ):
            _issue_once(issues, _cpp_specificity_issue())
        if (
            classification.topic_route == "cpp"
            and not _numeric_claims_supported_in_cpp_context(
                classification, retrieval, answer_draft
            )
        ):
            _issue_once(issues, _cpp_specificity_issue())
        if (
            classification.topic_route == "penal_substantivo"
            and not _penal_substantivo_matches_query(classification, confirmed)
        ):
            _issue_once(issues, _penal_gap_issue())

        if multi_article_noise_detected and (
            classification.needs_article_validation or not confirmed
        ):
            _issue_once(issues, _multi_article_issue())

        if classification.needs_article_validation and not confirmed:
            _issue_once(
                issues,
                _strict_gap_issue()
                if unsupported_articles
                else _weak_confirmation_issue(),
            )
        elif not confirmed and retrieval.official_evidence and prudential:
            _issue_once(issues, _weak_confirmation_issue())
        elif (
            not confirmed
            and not prudential
            and retrieval.official_evidence
            and classification.requires_strict_corpus_match
        ):
            _issue_once(issues, _strict_gap_issue())

        uses_general_knowledge = bool(answer_draft.rich_content)
        if uses_general_knowledge and (
            unsupported_articles
            or (classification.audience == "tecnico" and not confirmed)
            or (classification.main_branch == "misto" and not confirmed)
        ):
            _issue_once(
                issues,
                _general_knowledge_issue(
                    classification.needs_article_validation
                    or classification.audience == "tecnico"
                ),
            )

        if not retrieval.official_evidence:
            issues.append(
                ValidationIssue(
                    code="no_official_support",
                    message="A resposta não tem base oficial recuperada suficiente.",
                    severity="high",
                )
            )
        elif unsupported_articles or (
            classification.needs_article_validation and not confirmed
        ):
            _issue_once(issues, _limited_context_issue())

        confirmed = _limit_basis_for_route(classification, _dedupe_basis(confirmed))
        prudential = _limit_basis_for_route(classification, _dedupe_basis(prudential))
        is_technical = (
            classification.audience == "tecnico"
            or classification.needs_article_validation
        )
        sufficient_legal_support = (
            bool(retrieval.official_evidence)
            and not unsupported_articles
            and not retrieval.missing_branches
            and bool(confirmed or (prudential and not is_technical))
        )

        # Leniência para correcções: se o utilizador está a corrigir o sistema,
        # permitimos a resposta desde que haja suporte oficial, mesmo que a confirmação técnica seja parcial.
        if is_correction and retrieval.official_evidence:
            sufficient_legal_support = True

        if any(
            issue.code
            in {
                "penal_relevance_gap",
                "processual_specificity_gap",
                "branch_mismatch",
                "normative_conflict",
                "citator_gap",
                "vigency_unverified",
                "followup_anchor_unresolved",
            }
            for issue in issues
        ):
            sufficient_legal_support = False
        answer_mode = _answer_mode(
            classification, retrieval, issues, confirmed, sufficient_legal_support
        )

        return ValidationResult(
            confirmed_legal_basis=confirmed,
            prudential_legal_basis=prudential,
            issues=_compress_issues(issues),
            missing_branches=retrieval.missing_branches,
            unsupported_articles=unsupported_articles,
            source_cross_contamination=source_cross_contamination,
            sufficient_legal_support=sufficient_legal_support,
            official_sources_used=official_sources,
            user_sources_used=user_sources,
            answer_mode=answer_mode,
            jurisprudence_basis=jurisprudence_basis,
            normative_status=normative_status,
            normative_notes=normative_notes,
        )


legal_validation_service = LegalValidationService()
