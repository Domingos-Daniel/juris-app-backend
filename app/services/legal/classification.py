from __future__ import annotations

import json
import logging
import re
from typing import get_args

from app.services.legal.models import (
    AudienceType,
    LegalBranch,
    LegalClassification,
    NormTypeNeeded,
    RequestType,
    SpecificityLevel,
    TopicRoute,
)
from app.services.legal.pre_classifier import apply_pre_classification, pre_classify
from app.services.legal.prompts import ROUTER_SYSTEM_PROMPT
from app.services.llm.router import llm_router

logger = logging.getLogger(__name__)
ARTICLE_QUERY_RE = re.compile(r"(?:art|artigo|artigos)\s*(\d+[.]?\d*)", re.IGNORECASE)


def _robust_json_extract(raw: str) -> dict:
    """Extrai JSON de respostas LLM mesmo com wrap markdown ou texto extra."""
    if not raw:
        return {}
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _parse_classification_response(raw: str) -> dict:
    """Parse LLM classification response, handling json_object wrapper."""
    data = _robust_json_extract(raw)
    if not data:
        return {}
    if "json_object" in data and isinstance(data["json_object"], dict):
        data = data["json_object"]
    return data


_BRANCH_PARENTS: dict[str, str] = {
    "propriedade": "civil",
    "sucessorio": "civil",
}


def _expand_branch_candidates(main_branch: str) -> list[str]:
    if main_branch == "indeterminado" or not main_branch:
        return []
    candidates = [main_branch]
    parent = _BRANCH_PARENTS.get(main_branch)
    if parent and parent not in candidates:
        candidates.append(parent)
    return candidates


def _build_classification(
    question: str, data: dict, semantic_confidence: float = 0.0
) -> LegalClassification:
    """Build LegalClassification from LLM or semantic router data dict."""
    main_branch = data.get("main_branch")
    if main_branch not in get_args(LegalBranch):
        main_branch = "indeterminado"

    topic_route = data.get("topic_route")
    if topic_route not in get_args(TopicRoute):
        topic_route = "geral"

    request_type = data.get("request_type")
    if request_type not in get_args(RequestType):
        request_type = (
            "analise_tecnica"
            if request_type == "validacao_base_legal"
            else "explicacao_simples"
        )

    specificity = data.get("specificity")
    if specificity not in get_args(SpecificityLevel):
        specificity = "geral"

    audience = data.get("audience")
    if audience not in get_args(AudienceType):
        audience = "leigo"

    candidates = _expand_branch_candidates(main_branch)
    requested_article_numbers = [
        match.group(1).replace(".", "") for match in ARTICLE_QUERY_RE.finditer(question)
    ]
    specificity_value = data.get("specificity") or specificity
    needs_article_validation = specificity_value == "validacao_base_legal" or bool(
        requested_article_numbers
    )
    if requested_article_numbers and specificity == "geral":
        specificity = "validacao_base_legal"

    return LegalClassification(
        query_text=question,
        main_branch=main_branch,
        branch_candidates=candidates,
        request_type=request_type,
        specificity=specificity,
        audience=audience,
        is_follow_up=data.get("is_follow_up", False),
        is_correction=data.get("is_correction", False),
        is_transformation=data.get("is_transformation", False),
        transformation_type=data.get("transformation_type", "none"),
        topic_route=topic_route,
        search_query=data.get("search_query", question),
        norm_type_needed="misto",
        requires_strict_corpus_match=data.get("requires_strict_corpus_match", False),
        drafting_mode=request_type == "minuta_documental",
        explicit_branch_override=False,
        requested_article_numbers=requested_article_numbers,
        requested_diplomas=data.get("requested_diplomas", []),
        needs_article_validation=needs_article_validation,
        needs_source_separation=True,
        needs_practical_guidance=request_type
        in (
            "passos_praticos",
            "documentos_prova",
            "competencia_institucional",
            "estrategia_processual",
        ),
        needs_multi_branch_handling=main_branch == "misto"
        or specificity == "comparacao_multi_ramo",
        conversation_branch_hint=main_branch if data.get("is_follow_up") else None,
        conversation_topic_hint=topic_route if data.get("is_follow_up") else None,
        conversation_norm_type_hint=None,
        needs_clarification=data.get("needs_clarification", False),
        clarifying_questions=data.get("clarifying_questions", []),
        semantic_confidence=semantic_confidence,
    )


class LegalClassifier:
    async def classify(
        self,
        question: str,
        conversation_history: list[str] | None = None,
        provider: str | None = None,
    ) -> LegalClassification:
        # Try semantic router first (local, <50ms, no network)
        try:
            from app.services.rag.semantic_router import semantic_router

            semantic_branch, semantic_confidence = await semantic_router.classify(
                question
            )
        except Exception:
            semantic_branch, semantic_confidence = "indeterminado", 0.0

        # If semantic router is confident, use it and skip LLM classification
        if semantic_confidence >= 0.40 and semantic_branch != "indeterminado":
            logger.debug(
                "SemanticRouter: branch=%s confidence=%.2f — skipping LLM",
                semantic_branch,
                semantic_confidence,
            )
            data = {"main_branch": semantic_branch}
            data = apply_pre_classification(data, question)
            return _build_classification(question, data, semantic_confidence)

        # Fall back to LLM classifier for uncertain cases
        history_text = (
            "\n".join(conversation_history)
            if conversation_history
            else "Sem histórico."
        )
        prompt = (
            "Classifica a seguinte pergunta juridica angolana.\n\n"
            "DEVOLVE APENAS ESTE JSON (sem texto extra, sem markdown):\n"
            "{\n"
            '  "main_branch": "constitucional"|"civil"|"penal"|"laboral"|"administrativo"|"tributario"|"comercial"|"familia"|"propriedade"|"sucessorio"|"misto"|"indeterminado",\n'
            '  "topic_route": "constitucional"|"civil_obrigacoes"|"penal_substantivo"|"cpp"|"cpc"|"laboral"|"identificacao_civil"|"tributario"|"contencioso_admin"|"sociedades"|"familia"|"terras"|"sucessoes"|"geral",\n'
            '  "request_type": "explicacao_simples"|"analise_tecnica"|"passos_praticos",\n'
            '  "specificity": "geral"|"factual"|"follow_up",\n'
            '  "audience": "leigo"|"tecnico"|"misto",\n'
            '  "search_query": "melhores termos de pesquisa para o RAG",\n'
            '  "requires_strict_corpus_match": true|false,\n'
            '  "requested_diplomas": ["diploma"],\n'
            '  "is_follow_up": false,\n'
            '  "is_correction": false,\n'
            '  "is_transformation": false,\n'
            '  "transformation_type": "none",\n'
            '  "needs_clarification": false,\n'
            '  "clarifying_questions": []\n'
            "}\n\n"
            f"Historico:\n{history_text}\n\n"
            f"Pergunta: {question}"
        )

        # --- Pré-classificação determinística (zero custo, zero rede) ---
        pre_overrides = pre_classify(question)
        if pre_overrides:
            logger.debug("pre_classify: overrides detectados %s", pre_overrides)

        answer = ""
        try:
            answer, _ = await llm_router.generate(
                prompt,
                system_prompt=ROUTER_SYSTEM_PROMPT,
                json_mode=True,
                provider=provider,
            )
            data = _parse_classification_response(answer)
        except Exception:
            data = _robust_json_extract(answer) if answer else {}
            if not data:
                logger.warning(
                    "LegalClassifier: LLM nao retornou JSON parseavel. raw_preview=%s",
                    answer[:200],
                )

        data = apply_pre_classification(data, question)
        return _build_classification(question, data, semantic_confidence)


legal_classifier = LegalClassifier()
