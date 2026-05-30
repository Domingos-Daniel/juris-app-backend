from __future__ import annotations

import asyncio
import logging
import re
import sys
import time as _time
from dataclasses import asdict, replace

from cachetools import TTLCache
from app.core.config import get_settings
from app.db.models import ChatResponse, SourceItem
from app.db.postgres import postgres_manager

logger = logging.getLogger(__name__)

# Cache de classificações — evita chamar o LLM duas vezes (preflight + stream)
# Key: (query_normalized, tuple(history), provider)
# TTL: 5 minutos (classificações raramente mudam entre preflight e stream)
_classification_cache: TTLCache = TTLCache(maxsize=128, ttl=300)


def _is_legal_citation(content: str) -> bool:
    """Check if content looks like a legal citation (contains Art/artigo followed by number)."""
    stripped = content.strip()
    return bool(re.search(r"Art(?:igo|igos|\.)?\s*\d|[Aa]rtigo\s+\d", stripped))


def _looks_like_citation_ahead(text: str, max_dist: int = 80) -> bool:
    """Check if text within max_dist chars ahead looks like a citation start."""
    window = text[:max_dist] if len(text) > max_dist else text
    return bool(re.search(r"Art(?:igo|igos|\.)?\s*\d|[Aa]rtigo\s+\d", window))


def _find_close(text: str, start: int, pair: str) -> tuple[int, int] | None:
    """Find the closing bracket for a citation opened with pair ([[' or '((').

    Returns (close_idx, close_len) or None. Prefers balanced close (]] / ))),
    falls back to single close (] / )) if balanced not found nearby.
    """
    balanced_close = "]]" if pair == "[[" else "))"
    single_close = "]" if pair == "[[" else ")"

    # Try balanced close first — look within 220 chars (covers diploma names)
    search_end = min(len(text), start + 220)
    balanced_idx = text.find(balanced_close, start, search_end)
    if balanced_idx >= 0:
        content = text[start:balanced_idx]
        if _is_legal_citation(content):
            return (balanced_idx, 2)

    # Fall back to single close — look within 120 chars
    search_end = min(len(text), start + 120)
    single_idx = text.find(single_close, start, search_end)
    if single_idx >= 0:
        content = text[start:single_idx]
        if _is_legal_citation(content):
            return (single_idx, 1)

    return None


def _normalize_brackets(text: str) -> str:
    """Parse-based bracket normalizer.

    Handles both balanced [[...]]/((...)) and unbalanced [[...]/((...) patterns
    by preferring balanced close markers, falling back to single closes for
    unambiguous citation context.
    """
    if not text:
        return text
    text = text.replace("\x1f", "").replace("\x1e", "").replace("\x1d", "")
    text = text.replace("\x1c", "").replace("\x1b", "")

    result: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        if i + 1 < n:
            pair = text[i] + text[i + 1]

            if pair in ("[[", "(("):
                content_start = i + 2

                if _looks_like_citation_ahead(text[content_start:]):
                    close_info = _find_close(text, content_start, pair)

                    if close_info is not None:
                        close_idx, close_len = close_info
                        content = text[content_start:close_idx]
                        content = _normalize_brackets(content)
                        result.append("[")
                        result.append(content)
                        result.append("]")
                        i = close_idx + close_len
                        continue

        result.append(text[i])
        i += 1

    return "".join(result)


from app.services.legal import (
    legal_classifier,
    legal_composer,
    legal_confidence_service,
    legal_retrieval_service,
    legal_validation_service,
)
from app.services.legal.article_verifier import article_verifier
from app.services.llm.deepseek_client import deepseek_client
from app.services.legal.models import (
    ConfidenceResult,
    ValidationIssue,
    ValidationResult,
    RetrievalResult,
)
from app.services.legal.reranker import llm_reranker
from app.services.llm.router import llm_router
from app.services.rag.query_expander import query_expander
from app.services.rag.vector_store import legislation_vector_store


def _needs_clarification_from_classification(
    classification,
    semantic_confidence: float,
    has_follow_up_context: bool = False,
) -> bool:
    """Uses the classifier's own output metrics to detect genuinely vague queries.

    No hardcoded regex — relies on:
    1. Semantic router confidence (proxy for how well the query matches any known legal branch)
    2. main_branch (classifier's best guess at legal area)
    3. topic_route (specificity of sub-topic)
    4. Presence of requested diplomas or article numbers
    5. Quality of the generated search_query

    This gate acts as a safety net if the LLM classifier fails to set needs_clarification
    despite the prompt instructions.
    """
    # Follow-up context should not be treated as vague by default.
    # Short follow-up prompts like "fale mais" often have low semantic confidence,
    # but they are still answerable when there is conversation history/state.
    if has_follow_up_context and (
        classification.is_follow_up or classification.specificity == "follow_up"
    ):
        return False

    # Extreme low confidence: the query doesn't match any legal prototype.
    # Even if the LLM guessed a branch, it's unreliable — flag it.
    if semantic_confidence < 0.15:
        return True

    branch_unknown = classification.main_branch == "indeterminado"
    topic_generic = classification.topic_route == "geral"
    no_diplomas = not classification.requested_diplomas
    no_articles = not classification.requested_article_numbers

    if branch_unknown and topic_generic and no_diplomas and no_articles:
        if semantic_confidence < 0.3:
            return True
        if (
            not classification.search_query
            or len(classification.search_query.strip()) < 10
        ):
            return True

    # LLM guessed a branch but semantic confidence is still very low,
    # and the query has no structure (no diplomas, no articles, generic topic)
    # The LLM is guessing — don't trust it blindly
    if (
        semantic_confidence < 0.25
        and no_diplomas
        and no_articles
        and topic_generic
        and not classification.search_query
    ):
        return True

    return False


CLARIFYING_QUESTIONS_GENERAL = [
    "Qual e a area juridica principal do seu caso? (ex.: trabalho, familia, penal, fiscal)",
    "Qual e o facto concreto que aconteceu e o que pretende resolver?",
    "Se tiver, indique artigo, diploma ou entidade envolvida.",
]

CLARIFYING_QUESTIONS_FOLLOW_UP = [
    "Pretende que eu aprofunde a resposta anterior, compare com outra norma ou transforme em passos praticos?",
    "Qual e o ponto exacto que quer detalhar agora (artigo, prazo, procedimento, prova ou risco)?",
    "Quer manter o mesmo diploma da resposta anterior ou mudar para outro?",
]

SHORT_FOLLOW_UP_MARKERS = (
    "fale mais",
    "mais detalhes",
    "detalha",
    "detalhe",
    "explique melhor",
    "continua",
    "continue",
    "aprofunda",
    "aprofundar",
    "e depois",
    "e agora",
    "nesse caso",
    "no meu caso",
)


def _looks_short_follow_up_prompt(query: str) -> bool:
    normalized = (query or "").strip().casefold()
    if not normalized:
        return False
    if any(marker in normalized for marker in SHORT_FOLLOW_UP_MARKERS):
        return True
    return len(normalized.split()) <= 4 and normalized in {
        "sim",
        "e",
        "ok",
        "certo",
        "entendi",
        "pode continuar",
        "prossiga",
        "e o artigo",
    }


def _has_follow_up_context(
    history: list[str], classification, chat_state: dict | None
) -> bool:
    metadata = (chat_state or {}).get("metadata") or {}
    return bool(
        history
        or classification.is_follow_up
        or classification.specificity == "follow_up"
        or metadata.get("last_requested_article")
        or metadata.get("last_requested_diploma")
    )


def _default_clarifying_questions(
    query: str,
    classification,
    history: list[str],
    chat_state: dict | None,
) -> list[str]:
    if _has_follow_up_context(
        history, classification, chat_state
    ) and _looks_short_follow_up_prompt(query):
        return CLARIFYING_QUESTIONS_FOLLOW_UP
    return CLARIFYING_QUESTIONS_GENERAL


def _clarifying_message(query: str, history: list[str], classification) -> str:
    if history and _looks_short_follow_up_prompt(query):
        return (
            "Percebi que quer dar seguimento ao tema anterior. "
            "Para responder com rigor juridico e utilidade pratica, preciso de um detalhe adicional."
        )
    if classification.main_branch != "indeterminado":
        return "Tenho elementos iniciais, mas ainda falta um dado essencial para uma resposta juridica completa e bem fundamentada."
    return "Para lhe dar uma orientacao juridica precisa e fundamentada, preciso de 1-2 detalhes sobre o seu caso."


def _stabilize_follow_up_classification(
    query: str,
    history: list[str],
    classification,
    chat_state: dict | None,
):
    if (
        _has_follow_up_context(history, classification, chat_state)
        and _looks_short_follow_up_prompt(query)
        and not classification.is_follow_up
        and classification.specificity != "follow_up"
    ):
        return classification.model_copy(
            update={
                "is_follow_up": True,
                "specificity": "follow_up",
            }
        )
    return classification


# Separators that split multi-topic queries into sub-questions
# Matches: "E", "e", "e tambem", "alem disso", "bem como", "ou", "vs", "versus", "??"
MULTI_TOPIC_SEPARATORS = re.compile(
    r"(?:\s+[Ee]\s+(?:tamb[ée]m\s+)?)"
    r"|(?:\s+(?:al[ée]m\s+disso|bem\s+como|ou|vs\.?|versus)\s+)"
    r"|(?:\n{2,})"
    r"|(?:\?\s+(?:[A-Za-zÀ-ÿ]))",
)


def _detect_multi_topic(query: str) -> bool:
    """Detect queries with multiple distinct legal topics.

    Uses separator-based splitting with validation:
    - Both parts must have >= 3 words (substantial enough to be a topic)
    - Need at least 2 substantial sub-questions
    """
    parts = MULTI_TOPIC_SEPARATORS.split(query)
    if len(parts) < 2:
        return False

    substantial = 0
    for part in parts:
        words = part.strip().split()
        if len(words) >= 2 and any(len(w) > 2 for w in words):
            substantial += 1

    return substantial >= 2


REQUESTED_DIPLOMA_SLUGS = {
    "Lei Geral do Trabalho": "lei-geral-do-trabalho-lei-12-23",
    "Código Penal": "codigo-penal-lei-38-20",
    "Codigo Penal": "codigo-penal-lei-38-20",
    "Código Civil": "codigo-civil",
    "Codigo Civil": "codigo-civil",
    "Constituição da República de Angola": "constituicao-republica-angola-2022",
    "Constituicao da Republica de Angola": "constituicao-republica-angola-2022",
    "Código do Processo Penal": "codigo-processo-penal-lei-39-20",
    "Codigo do Processo Penal": "codigo-processo-penal-lei-39-20",
    "Lei do Contencioso Administrativo": "codigo-processo-contencioso-administrativo-33-22",
    "Lei do Bilhete de Identidade": "lei-bilhete-identidade-4-09",
    "Lei das Sociedades Comerciais": "lei-sociedades-comerciais-1-04",
    "Código Geral Tributário": "codigo-geral-tributario-21-14",
    "Codigo Geral Tributario": "codigo-geral-tributario-21-14",
    "Código de Família": "codigo-familia-lei-1-88",
    "Codigo de Familia": "codigo-familia-lei-1-88",
    "Lei de Terras": "lei-terras-9-04",
}

DIPLOMA_TITLE_HINTS = {
    "lei geral do trabalho": "lei-geral-do-trabalho-lei-12-23",
    "codigo penal": "codigo-penal-lei-38-20",
    "código penal": "codigo-penal-lei-38-20",
    "codigo civil": "codigo-civil",
    "código civil": "codigo-civil",
    "constituicao da republica de angola": "constituicao-republica-angola-2022",
    "constituição da república de angola": "constituicao-republica-angola-2022",
    "codigo do processo penal": "codigo-processo-penal-lei-39-20",
    "código do processo penal": "codigo-processo-penal-lei-39-20",
    "lei do contencioso administrativo": "codigo-processo-contencioso-administrativo-33-22",
    "lei do bilhete de identidade": "lei-bilhete-identidade-4-09",
    "lei das sociedades comerciais": "lei-sociedades-comerciais-1-04",
    "codigo geral tributario": "codigo-geral-tributario-21-14",
    "código geral tributário": "codigo-geral-tributario-21-14",
    "codigo de familia": "codigo-familia-lei-1-88",
    "código de família": "codigo-familia-lei-1-88",
    "lei de terras": "lei-terras-9-04",
}

SLUG_TO_DIPLOMA_NAME = {
    "lei-geral-do-trabalho-lei-12-23": "Lei Geral do Trabalho",
    "codigo-penal-lei-38-20": "Código Penal",
    "codigo-civil": "Código Civil",
    "constituicao-republica-angola-2022": "Constituição da República de Angola",
    "codigo-processo-penal-lei-39-20": "Código do Processo Penal",
    "codigo-processo-contencioso-administrativo-33-22": "Lei do Contencioso Administrativo",
    "lei-bilhete-identidade-4-09": "Lei do Bilhete de Identidade",
    "lei-sociedades-comerciais-1-04": "Lei das Sociedades Comerciais",
    "codigo-geral-tributario-21-14": "Código Geral Tributário",
    "codigo-familia-lei-1-88": "Código de Família",
    "lei-terras-9-04": "Lei de Terras",
}
FOLLOW_UP_REFERENCE_MARKERS = (
    "esse mesmo artigo",
    "esse artigo",
    "o mesmo artigo",
    "mesmo artigo",
    "essa mesma norma",
    "essa norma",
    "esse diploma",
    "o mesmo diploma",
    "esse mesmo diploma",
)
FOLLOW_UP_DIPLOMA_MARKERS = (
    "esse diploma",
    "o mesmo diploma",
    "essa lei",
    "essa constituicao",
    "essa constituição",
)

ARTICLE_QUERY_RE = re.compile(r"(?:art|artigo|artigos)\s*(\d+[.]?\d*)", re.IGNORECASE)


class RAGPipeline:
    @staticmethod
    def _normalize_text(text: str) -> str:
        return (text or "").strip().casefold()

    @staticmethod
    def _recent_user_questions(history: list[str]) -> list[str]:
        return [
            item.split(":", 1)[1].strip()
            for item in history
            if item.lower().startswith("utilizador:") and ":" in item
        ]

    @staticmethod
    def _extract_articles_from_text(text: str) -> list[str]:
        return [
            match.group(1).replace(".", "")
            for match in ARTICLE_QUERY_RE.finditer(text or "")
        ]

    @staticmethod
    def _history_anchor_diploma(history: list[str]) -> str | None:
        user_questions = RAGPipeline._recent_user_questions(history)
        for item in reversed(user_questions):
            normalized = RAGPipeline._normalize_text(item)
            for diploma_name in REQUESTED_DIPLOMA_SLUGS:
                if diploma_name.casefold() in normalized:
                    return diploma_name
        return None

    @staticmethod
    def _history_anchor_article(history: list[str]) -> str | None:
        user_questions = RAGPipeline._recent_user_questions(history)
        for item in reversed(user_questions):
            matches = RAGPipeline._extract_articles_from_text(item)
            if matches:
                return matches[0]
        return None

    @staticmethod
    def _looks_referential_follow_up(query: str, classification) -> bool:
        normalized = RAGPipeline._normalize_text(query)
        if (
            classification.specificity != "follow_up"
            and not classification.is_follow_up
        ):
            return False
        return any(marker in normalized for marker in FOLLOW_UP_REFERENCE_MARKERS)

    @staticmethod
    def _hydrate_follow_up_context(
        query: str,
        history: list[str],
        classification,
        chat_state: dict | None,
    ):
        if not RAGPipeline._looks_referential_follow_up(query, classification):
            return classification

        metadata = (chat_state or {}).get("metadata") or {}
        anchor_article = (
            (chat_state or {}).get("active_article")
            or metadata.get("unresolved_requested_article")
            or metadata.get("last_requested_article")
            or RAGPipeline._history_anchor_article(history)
        )
        anchor_slug = (
            (chat_state or {}).get("diploma_slug")
            or metadata.get("active_diploma_slug")
            or metadata.get("last_requested_diploma_slug")
        )
        anchor_diploma = (
            SLUG_TO_DIPLOMA_NAME.get(anchor_slug or "")
            or metadata.get("last_requested_diploma")
            or RAGPipeline._history_anchor_diploma(history)
        )

        requested_articles = list(classification.requested_article_numbers)
        if anchor_article and anchor_article not in requested_articles:
            requested_articles.append(anchor_article)

        requested_diplomas = list(classification.requested_diplomas)
        if anchor_diploma and anchor_diploma not in requested_diplomas:
            requested_diplomas.append(anchor_diploma)

        requires_strict = classification.requires_strict_corpus_match
        if requested_articles or any(
            marker in RAGPipeline._normalize_text(query)
            for marker in FOLLOW_UP_DIPLOMA_MARKERS
        ):
            requires_strict = True

        return classification.model_copy(
            update={
                "requested_article_numbers": requested_articles,
                "requested_diplomas": requested_diplomas,
                "needs_article_validation": bool(requested_articles)
                or classification.needs_article_validation,
                "requires_strict_corpus_match": requires_strict,
            }
        )

    @staticmethod
    async def _classify_query(query: str, history: list[str], provider: str | None):
        cache_key = (query.strip().casefold(), tuple(history or ()), provider)
        cached = _classification_cache.get(cache_key)
        if cached is not None:
            logger.debug("Classification cache HIT for: %s", query[:60])
            return cached
        try:
            result = await legal_classifier.classify(query, history, provider=provider)
        except TypeError:
            result = await legal_classifier.classify(query, history)
        _classification_cache[cache_key] = result
        return result

    @staticmethod
    def _derive_search_query(query: str, history: list[str], classification) -> str:
        normalized_query = (query or "").strip()

        # Transformation follow-ups ("resuma", "fale mais") —
        # anchor the retrieval on the last substantive user question, not the command itself.
        if classification.is_transformation and history:
            for item in reversed(history):
                if item.startswith("Utilizador:") and not any(
                    m in item.casefold()
                    for m in (
                        "resum",
                        "fale mais",
                        "simplif",
                        "continue",
                        "explique",
                        "detalh",
                        "aprofund",
                        "traduz",
                    )
                ):
                    return item.replace("Utilizador:", "").strip()
            return normalized_query

        article_matches = [
            match.group(1).replace(".", "")
            for match in ARTICLE_QUERY_RE.finditer(normalized_query)
        ]
        if not article_matches and classification.requested_article_numbers:
            article_matches = list(classification.requested_article_numbers)

        if article_matches and classification.requested_diplomas:
            return f"artigo {article_matches[0]} {classification.requested_diplomas[0]}"

        if article_matches:
            return f"artigo {article_matches[0]}"

        short_follow_up = len(normalized_query.split()) < 5 and history
        if not short_follow_up:
            return normalized_query

        if re.fullmatch(r"[Ee]?\s*o?\s*\d+[?!.]?", normalized_query):
            number_match = re.search(r"\d+[.]?\d*", normalized_query)
            if number_match:
                article_number = number_match.group(0).replace(".", "")
                diploma = (
                    classification.requested_diplomas[0]
                    if classification.requested_diplomas
                    else ""
                )
                base_query = f"artigo {article_number}"
                return f"{base_query} {diploma}".strip()

        for item in reversed(history):
            if item.startswith("Utilizador:"):
                return item.replace("Utilizador:", "").strip()
        return normalized_query

    @staticmethod
    def _basis_slug_from_retrieval(item, retrieval) -> str:
        for evidence in retrieval.official_evidence + retrieval.user_evidence:
            chunk = evidence.chunk
            if (
                chunk.title == item.diploma
                and chunk.page == item.page
                and chunk.source_scope == item.source_scope
            ):
                metadata = chunk.metadata or {}
                slug = metadata.get("diploma_slug")
                if slug:
                    return slug

        diploma_key = (
            re.sub(r"\s*\(.*?\)\s*$", "", (item.diploma or "")).strip().lower()
        )
        return DIPLOMA_TITLE_HINTS.get(diploma_key, "")

    async def answer_query(
        self,
        query: str,
        provider: str | None = None,
        conversation_history: list[str] | None = None,
        chat_id: str | None = None,
        active_document_id: str | None = None,
        user_id: str | int | None = None,
    ) -> ChatResponse:
        normalized_query = (query or "").strip()
        if not normalized_query:
            raise ValueError("A pergunta não pode estar vazia.")

        current_chat_id = chat_id
        history = conversation_history or []
        provider_used = provider or get_settings().default_llm_provider
        chat_state = (
            postgres_manager.get_conversation_state(current_chat_id, user_id=user_id)
            if current_chat_id
            else None
        )

        classification = await self._classify_query(normalized_query, history, provider)
        classification = self._hydrate_follow_up_context(
            normalized_query, history, classification, chat_state
        )
        classification = _stabilize_follow_up_classification(
            normalized_query, history, classification, chat_state
        )

        # Detect very vague or off-topic queries using classifier's own output metrics
        if (
            not classification.needs_clarification
            and not classification.clarifying_questions
        ):
            if _needs_clarification_from_classification(
                classification,
                classification.semantic_confidence,
                has_follow_up_context=_has_follow_up_context(
                    history, classification, chat_state
                ),
            ):
                classification.needs_clarification = True
                classification.clarifying_questions = _default_clarifying_questions(
                    normalized_query, classification, history, chat_state
                )

        # Multi-topic detection: force multi-branch when query has "E" separating topics
        if (
            not classification.needs_multi_branch_handling
            and classification.main_branch != "misto"
            and _detect_multi_topic(normalized_query)
        ):
            classification.needs_multi_branch_handling = True
            # Expand branch candidates: add pre-classifier + keyword detection
            from app.services.legal.pre_classifier import pre_classify

            pre = pre_classify(normalized_query)
            pre_branch = pre.get("main_branch") if pre else None
            if pre_branch and pre_branch not in classification.branch_candidates:
                classification.branch_candidates = list(
                    classification.branch_candidates
                ) + [pre_branch]

            # Keyword-based branch detection for branches not covered by pre-classifier
            branch_kw = {
                "comercial": r"s[oó]cio|sociedad|quotas?|societ[aá]rio|accionista|delibera[cç][aã]o",
                "civil": r"contrato|obriga[cç][aã]o|responsabilidade civil|indemniza[cç][aã]o|arrendamento",
                "penal": r"crime|pena|pris[aã]o|furto|roubo|homic[ií]dio|burla",
                "tributario": r"imposto|IVA|IRC|reten[cç][aã]o na fonte",
                "familia": r"div[oó]rcio|casamento|filhos?|alimentos|paternidade|ado[cç][aã]o",
                "constitucional": r"constitui[cç][aã]o|direitos fundamentais|liberdade",
                "administrativo": r"funcion[aá]rio|acto administrativo|concurso p[uú]blico|licen[cç]a",
            }
            q_lower = normalized_query.lower()
            for branch, pat in branch_kw.items():
                if branch in classification.branch_candidates:
                    continue
                if re.search(pat, q_lower):
                    classification.branch_candidates = list(
                        classification.branch_candidates
                    ) + [branch]
                    break

            # Inject secondary branch diplomas into requested_diplomas
            from app.services.legal.retrieval import BRANCH_DIPLOMAS as _BD

            for branch in classification.branch_candidates:
                if branch == classification.main_branch:
                    continue
                branch_diplomas = _BD.get(branch, tuple())
                for diploma_name in branch_diplomas:
                    if diploma_name not in classification.requested_diplomas:
                        classification.requested_diplomas = list(
                            classification.requested_diplomas
                        ) + [diploma_name]
                        break
                break

        # Follow-up questions with short queries — use latest user question for retrieval
        search_query = self._derive_search_query(
            normalized_query, history, classification
        )
        classification.query_text = search_query

        if classification.needs_clarification and classification.clarifying_questions:
            clarifying_answer = _clarifying_message(
                normalized_query, history, classification
            )
            if not current_chat_id:
                current_chat_id = postgres_manager.create_chat(
                    title=normalized_query,
                    active_document_id=active_document_id,
                    user_id=user_id,
                )
            postgres_manager.save_query(
                question=normalized_query, answer=clarifying_answer
            )
            return ChatResponse(
                answer=clarifying_answer,
                sources=[],
                provider_used=provider_used,
                chat_id=current_chat_id,
                active_document_id=active_document_id,
                answer_mode="clarifying",
                classification=classification.model_dump(),
                clarifying_questions=classification.clarifying_questions,
            )

        if (
            classification.requires_strict_corpus_match
            and classification.requested_diplomas
        ):
            available_slugs = legislation_vector_store.available_diploma_slugs()
            requested_slugs = {
                REQUESTED_DIPLOMA_SLUGS[diploma]
                for diploma in classification.requested_diplomas
                if diploma in REQUESTED_DIPLOMA_SLUGS
            }
            if requested_slugs and not requested_slugs.intersection(available_slugs):
                answer = (
                    "O tema pedido ainda não está coberto de forma suficiente no corpus jurídico público actualmente indexado. "
                    "A resposta segura para esta rota exige que o diploma prioritário esteja carregado e validado localmente."
                )
                if not current_chat_id:
                    current_chat_id = postgres_manager.create_chat(
                        title=normalized_query,
                        active_document_id=active_document_id,
                        user_id=user_id,
                    )
                postgres_manager.append_chat_exchange(
                    chat_id=current_chat_id,
                    question=normalized_query,
                    answer=answer,
                    provider_used=provider_used,
                    sources=[],
                    active_document_id=active_document_id,
                )
                postgres_manager.save_query(question=normalized_query, answer=answer)
                return ChatResponse(
                    answer=answer,
                    sources=[],
                    provider_used=provider_used,
                    chat_id=current_chat_id,
                    active_document_id=active_document_id,
                    answer_mode="refused",
                    confidence={
                        "level": "baixa",
                        "score": 0.0,
                        "reasons": [
                            "O diploma prioritário ainda não está disponível no corpus indexado."
                        ],
                    },
                    classification=classification.model_dump(),
                    legal_basis=[],
                    validation_issues=[
                        {
                            "code": "corpus_gap",
                            "message": "O diploma prioritário desta rota ainda não está disponível no corpus indexado.",
                            "severity": "high",
                        }
                    ],
                )
        # Se for uma transformação (ex: "diz em termos simples"), podemos reutilizar o contexto do histórico
        # ou apenas permitir que o LLM processe a resposta anterior sem exigir novos documentos.
        if active_document_id and classification.is_transformation:
            classification = classification.model_copy(
                update={"is_transformation": False}
            )
        is_transformation = classification.is_transformation

        if is_transformation:
            retrieval = RetrievalResult(
                classification=classification,
                official_evidence=[],
                user_evidence=[],
                missing_branches=[],
            )
        else:
            expanded_queries = query_expander.expand(search_query, classification)
            results = await asyncio.gather(
                *[
                    legal_retrieval_service.retrieve(
                        qv,
                        classification,
                        conversation_history=history,
                        active_document_id=active_document_id,
                    )
                    for qv in expanded_queries[:1]
                ]
            )
            all_evidences: list = []
            all_user_evidences: list = []
            for partial in results:
                all_evidences.extend(partial.official_evidence)
                all_user_evidences.extend(partial.user_evidence)

            if all_evidences:
                all_evidences = sorted(
                    all_evidences, key=lambda e: e.score, reverse=True
                )[:12]
            combined_evidences = sorted(
                all_evidences + all_user_evidences, key=lambda e: e.score, reverse=True
            )[:12]
            if combined_evidences:
                retrieval = RetrievalResult(
                    classification=classification,
                    official_evidence=all_evidences,
                    user_evidence=all_user_evidences,
                    missing_branches=[],
                    retrieved_chunks=[e.chunk for e in combined_evidences],
                )
            else:
                retrieval = await legal_retrieval_service.retrieve(
                    classification.search_query or normalized_query,
                    classification,
                    conversation_history=history,
                    active_document_id=active_document_id,
                )

            retrieval = self._filter_retrieval_by_branch(retrieval, classification)

            if len(retrieval.retrieved_chunks) > 10:
                chunk_texts = [chunk.text or "" for chunk in retrieval.retrieved_chunks]
                relevance = await llm_reranker.rerank(
                    normalized_query,
                    chunk_texts,
                    provider=provider,
                )
                min_len = min(len(retrieval.official_evidence), len(relevance))
                retrieval.official_evidence = [
                    retrieval.official_evidence[i]
                    for i in range(min_len)
                    if relevance[i]
                ]
                if not retrieval.official_evidence and min_len > 0:
                    retrieval.official_evidence = retrieval.official_evidence[:5]
                chunks = []
                seen = set()
                for ev in retrieval.official_evidence + retrieval.user_evidence:
                    cid = id(ev.chunk)
                    if cid not in seen:
                        seen.add(cid)
                        chunks.append(ev.chunk)
                retrieval.retrieved_chunks = chunks

        if not retrieval.retrieved_chunks and not is_transformation:
            answer = (
                "Não encontrei contexto jurídico suficiente no índice actual para responder com segurança. "
                "Reformule a pergunta com mais detalhe, indique o diploma pretendido ou peça o artigo exacto a confirmar."
            )
            if not current_chat_id:
                current_chat_id = postgres_manager.create_chat(
                    title=normalized_query,
                    active_document_id=active_document_id,
                    user_id=user_id,
                )
            postgres_manager.append_chat_exchange(
                chat_id=current_chat_id,
                question=normalized_query,
                answer=answer,
                provider_used=provider_used,
                sources=[],
                active_document_id=active_document_id,
            )
            postgres_manager.save_query(question=normalized_query, answer=answer)
            return ChatResponse(
                answer=answer,
                sources=[],
                provider_used=provider_used,
                chat_id=current_chat_id,
                active_document_id=active_document_id,
                answer_mode="refused",
                confidence={
                    "level": "baixa",
                    "score": 0.0,
                    "reasons": [
                        "Sem contexto jurídico recuperado suficiente para uma resposta verificável."
                    ],
                },
                classification=classification.model_dump(),
                legal_basis=[],
                validation_issues=[
                    {
                        "code": "no_context",
                        "message": "Sem contexto jurídico suficiente no índice actual.",
                        "severity": "high",
                    }
                ],
            )

        prompt = legal_composer.build_prompt(
            normalized_query,
            classification,
            retrieval.retrieved_chunks,
            conversation_history=history,
        )
        _tllm = _time.time()
        try:
            raw_answer, provider_used = await llm_router.generate(
                prompt,
                provider=provider,
                max_tokens=800 if classification.audience == "leigo" else 1200,
            )
        except RuntimeError as exc:
            answer_text = str(exc)
            if not current_chat_id:
                current_chat_id = postgres_manager.create_chat(
                    title=normalized_query,
                    active_document_id=active_document_id,
                    user_id=user_id,
                )
            postgres_manager.append_chat_exchange(
                chat_id=current_chat_id,
                question=normalized_query,
                answer=answer_text,
                provider_used=provider_used,
                sources=[],
                active_document_id=active_document_id,
            )
            postgres_manager.save_query(question=normalized_query, answer=answer_text)
            return ChatResponse(
                answer=answer_text,
                sources=[],
                provider_used=provider_used,
                chat_id=current_chat_id,
                active_document_id=active_document_id,
                answer_mode="refused",
                confidence={
                    "level": "baixa",
                    "score": 0.0,
                    "reasons": [answer_text],
                },
                classification=classification.model_dump(),
                legal_basis=[],
                validation_issues=[
                    {
                        "code": "llm_unavailable",
                        "message": answer_text,
                        "severity": "high",
                    }
                ],
            )
        _tllm = _time.time() - _tllm
        _tpp = _time.time()
        answer_draft = legal_composer.parse_llm_json(raw_answer)
        answer_draft = legal_composer.constrain_draft_to_context(
            answer_draft, retrieval.retrieved_chunks
        )
        validation = legal_validation_service.validate(
            classification, retrieval, answer_draft
        )

        # Em transformações, forçamos o modo grounded se o LLM conseguiu gerar algo útil,
        # ignorando a falta de novos chunks oficiais.
        if is_transformation and answer_draft.rich_content:
            validation.sufficient_legal_support = True
            validation.answer_mode = "grounded"
            validation.issues = []

        if classification.topic_route == "cpp" and validation.answer_mode == "grounded":
            answer_numbers = {
                part
                for part in re.findall(
                    r"\b\d+[.,]?\d*\b", answer_draft.rich_content or ""
                )
            }
            context_text = " ".join(
                (evidence.chunk.text or "") for evidence in retrieval.official_evidence
            ).lower()
            if answer_numbers and not any(
                number in context_text for number in answer_numbers
            ):
                issues = list(validation.issues)
                if not any(
                    issue.code == "processual_specificity_gap" for issue in issues
                ):
                    issues.append(
                        ValidationIssue(
                            code="processual_specificity_gap",
                            message="A base processual recuperada ainda não confirma com precisão suficiente o ponto específico perguntado sobre prisão preventiva no CPP.",
                            severity="medium",
                        )
                    )
                validation = validation.model_copy(
                    update={
                        "answer_mode": "limited",
                        "sufficient_legal_support": False,
                        "issues": issues,
                    }
                )
                answer_draft = legal_composer.fallback_from_validation(
                    validation, original_draft=answer_draft
                )
        cpp_answer_text = (answer_draft.rich_content or "").lower()
        if classification.topic_route == "cpp" and validation.answer_mode == "grounded":
            if any(
                marker in cpp_answer_text
                for marker in (
                    "não é explicitamente mencionado",
                    "nao e explicitamente mencionado",
                    "não menciona",
                    "nao menciona",
                )
            ):
                issues = list(validation.issues)
                if not any(
                    issue.code == "processual_specificity_gap" for issue in issues
                ):
                    issues.append(
                        ValidationIssue(
                            code="processual_specificity_gap",
                            message="A base processual recuperada ainda não confirma com precisão suficiente o ponto específico perguntado sobre prisão preventiva no CPP.",
                            severity="medium",
                        )
                    )
                validation = validation.model_copy(
                    update={
                        "answer_mode": "limited",
                        "sufficient_legal_support": False,
                        "issues": issues,
                    }
                )
        if not is_transformation:
            if not legal_composer.answer_tracks_question(
                normalized_query, answer_draft, classification
            ):
                answer_draft = legal_composer.fallback_from_validation(
                    validation, original_draft=answer_draft
                )
                if validation.answer_mode == "grounded":
                    validation = validation.model_copy(
                        update={"answer_mode": "limited"}
                    )
            if legal_composer.answer_looks_like_json_artifact(
                answer_draft.rich_content
            ):
                answer_draft = legal_composer.fallback_from_validation(
                    validation, original_draft=answer_draft
                )
        verified_articles = []
        to_verify = [
            (
                item.article or "",
                self._basis_slug_from_retrieval(item, retrieval),
                item.page,
            )
            for item in validation.confirmed_legal_basis
            + validation.prudential_legal_basis
            if item.article and self._basis_slug_from_retrieval(item, retrieval)
        ]
        if to_verify:
            try:
                verified_articles = await asyncio.wait_for(
                    article_verifier.verify_batch(to_verify), timeout=1.5
                )
            except asyncio.TimeoutError:
                verified_articles = []
                logger.info("Article verification timed out — proceeding without it")
            unverified = [va for va in verified_articles if va.status == "not_found"]
            has_unsupported = any(
                issue.code == "unsupported_article" for issue in validation.issues
            )
            if (
                unverified
                and not has_unsupported
                and validation.answer_mode != "grounded_with_caveat"
            ):
                evidence_count = len(retrieval.official_evidence)
                severity = (
                    "medium"
                    if validation.answer_mode == "grounded" and evidence_count <= 2
                    else "high"
                )
                validation.issues.append(
                    ValidationIssue(
                        code="unverified_article",
                        message=f"{len(unverified)} artigo(s) citados não foram confirmados no corpus indexado.",
                        severity=severity,
                    )
                )
            if (
                classification.needs_article_validation
                or classification.requires_strict_corpus_match
            ) and any(
                va.status not in {"confirmed", "confirmed_in_text"}
                for va in verified_articles
            ):
                validation = validation.model_copy(
                    update={
                        "answer_mode": "limited",
                        "sufficient_legal_support": False,
                        "issues": validation.issues,
                    }
                )
                answer_draft = legal_composer.fallback_from_validation(
                    validation, original_draft=None
                )

        requested_articles = list(classification.requested_article_numbers)
        if not requested_articles:
            requested_articles = [
                match.group(1).replace(".", "")
                for match in ARTICLE_QUERY_RE.finditer(classification.query_text or "")
            ]
        confirmed_articles = {
            va.article.replace(".", "")
            for va in verified_articles
            if va.status in {"confirmed", "confirmed_in_text"} and va.article
        }
        if requested_articles and not confirmed_articles.intersection(
            requested_articles
        ):
            validation.issues.append(
                ValidationIssue(
                    code="requested_article_not_recovered",
                    message="O artigo exacto pedido não foi confirmado no contexto recuperado.",
                    severity="high"
                    if classification.requires_strict_corpus_match
                    else "medium",
                )
            )
            validation = validation.model_copy(
                update={
                    "answer_mode": "limited",
                    "sufficient_legal_support": False,
                    "issues": validation.issues,
                }
            )
            answer_draft = legal_composer.fallback_from_validation(
                validation, original_draft=None
            )

        referential_follow_up = self._looks_referential_follow_up(
            normalized_query, classification
        )
        unresolved_anchor = bool(
            referential_follow_up
            and requested_articles
            and not confirmed_articles.intersection(requested_articles)
        )
        if unresolved_anchor and not any(
            issue.code == "followup_anchor_unresolved" for issue in validation.issues
        ):
            validation.issues.append(
                ValidationIssue(
                    code="followup_anchor_unresolved",
                    message="O follow-up refere-se ao mesmo artigo anterior, mas esse artigo ainda não foi confirmado no contexto recuperado.",
                    severity="high",
                )
            )
            validation = validation.model_copy(
                update={
                    "answer_mode": "limited",
                    "sufficient_legal_support": False,
                    "issues": validation.issues,
                }
            )
            answer_draft = legal_composer.fallback_from_validation(
                validation, original_draft=None
            )

        answer_haystack = (answer_draft.rich_content or "").lower()
        if any(
            marker in answer_haystack
            for marker in (
                "não contém",
                "nao contem",
                "não consta",
                "nao consta",
                "não especifica",
                "nao especifica",
                "não é possível",
                "nao e possivel",
                "não foi possível",
                "nao foi possivel",
            )
        ):
            validation = validation.model_copy(
                update={
                    "answer_mode": "limited",
                    "sufficient_legal_support": False,
                }
            )

        confidence = legal_confidence_service.score(
            classification, retrieval, validation, verified_articles
        )
        sources = self._select_sources(retrieval, validation)
        # Force user document sources when a document is active
        if active_document_id and not any(
            s.source_scope == "user_upload" for s in sources
        ):
            logger.info(
                "Force-adding user doc sources (non-stream) for %s",
                active_document_id[:8],
            )
            try:
                for ev in retrieval.user_evidence:
                    chunk = ev.chunk
                    sources.append(
                        SourceItem(
                            title=chunk.title,
                            source=chunk.source,
                            link_original=chunk.link_original,
                            deep_link=(
                                f"{chunk.link_original}#page={chunk.page}"
                                if chunk.link_original
                                and chunk.page
                                and "#page=" not in chunk.link_original
                                else chunk.link_original
                            ),
                            page=chunk.page,
                            article_number=chunk.article_number,
                            law_status=chunk.law_status,
                            excerpt=chunk.text[:780],
                            attribution_text=chunk.text[:300] if chunk.text else None,
                            source_scope=chunk.source_scope,
                            document_id=chunk.document_id,
                        )
                    )
            except Exception as exc:
                logger.warning("Force-add sources (non-stream) failed: %s", exc)
        answer = legal_composer.compose_answer(
            classification, answer_draft, validation, confidence, sources
        )
        answer = legal_composer.sanitize_answer(answer)
        answer = _normalize_brackets(answer)
        _tpp = _time.time() - _tpp
        logger.info("LLM:%.1fs postproc:%.1fs", _tllm, _tpp)

        if answer.startswith("{"):
            import json

            try:
                data = json.loads(answer)
                if isinstance(data, dict):
                    for key in ("rich_content", "answer", "response", "direct_answer"):
                        if key in data and isinstance(data[key], str):
                            answer = data[key]
                            break
            except Exception:
                extracted = legal_composer._extract_rich_content(answer)
                if extracted:
                    answer = extracted
                else:
                    for key in ("rich_content", "direct_answer", "simple_explanation"):
                        m = re.search(
                            rf'"{key}"\s*:\s*"((?:(?:\\.)|[^"\\])*)', answer, re.DOTALL
                        )
                        if m:
                            try:
                                answer = (
                                    m.group(1)
                                    .encode()
                                    .decode("unicode_escape", errors="replace")
                                )
                            except Exception:
                                answer = m.group(1)
                            break
                    if answer.startswith("{") or len(answer) < 50:
                        cleaned = re.sub(
                            r'^\{.*?"rich_content"\s*:\s*"', "", answer, flags=re.DOTALL
                        )
                        cleaned = re.sub(
                            r'",\s*"cited_.*$', "", cleaned, flags=re.DOTALL
                        )
                        cleaned = re.sub(r"\\n", "\n", cleaned)
                        if len(cleaned) > 50:
                            answer = cleaned

        if "```json" in answer:
            answer = re.sub(r"```json\s*\{.*?\}\s*```", "", answer, flags=re.DOTALL)

        answer = _normalize_brackets(answer)

        if not current_chat_id:
            current_chat_id = postgres_manager.create_chat(
                title=normalized_query,
                active_document_id=active_document_id,
                user_id=user_id,
            )
        postgres_manager.append_chat_exchange(
            chat_id=current_chat_id,
            question=normalized_query,
            answer=answer,
            provider_used=provider_used,
            sources=[source.model_dump() for source in sources],
            active_document_id=active_document_id,
        )
        postgres_manager.save_query(question=normalized_query, answer=answer)

        primary_basis = (
            validation.confirmed_legal_basis[:1]
            or validation.prudential_legal_basis[:1]
            or validation.jurisprudence_basis[:1]
        )
        primary_slug = (
            self._basis_slug_from_retrieval(primary_basis[0], retrieval)
            if primary_basis
            else (
                REQUESTED_DIPLOMA_SLUGS.get(classification.requested_diplomas[0])
                if classification.requested_diplomas
                else None
            )
        )
        state_article = (
            requested_articles[0]
            if requested_articles
            else (next(iter(confirmed_articles)) if confirmed_articles else None)
        )
        postgres_manager.upsert_conversation_state(
            chat_id=current_chat_id,
            user_id=user_id,
            topic_route=classification.topic_route,
            legal_branch=classification.main_branch,
            diploma_slug=primary_slug,
            active_article=state_article,
            metadata={
                "last_requested_article": requested_articles[0]
                if requested_articles
                else None,
                "last_requested_diploma": classification.requested_diplomas[0]
                if classification.requested_diplomas
                else None,
                "last_requested_diploma_slug": primary_slug,
                "last_answer_mode": validation.answer_mode,
                "last_issue_codes": [issue.code for issue in validation.issues],
                "unresolved_requested_article": (
                    requested_articles[0]
                    if requested_articles
                    and not confirmed_articles.intersection(requested_articles)
                    else None
                ),
                "active_diploma_slug": primary_slug,
                "normative_status": validation.normative_status,
            },
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            provider_used=provider_used,
            chat_id=current_chat_id,
            active_document_id=active_document_id,
            answer_mode=validation.answer_mode,
            confidence=confidence.model_dump(),
            classification=classification.model_dump(),
            legal_basis=[
                item.model_dump()
                for item in validation.confirmed_legal_basis
                + validation.prudential_legal_basis
                + validation.jurisprudence_basis
            ],
            validation_issues=[issue.model_dump() for issue in validation.issues],
            verified_articles=[asdict(va) for va in verified_articles],
        )

    async def preflight_classify(
        self,
        query: str,
        provider: str | None = None,
        conversation_history: list[str] | None = None,
        chat_id: str | None = None,
        user_id: str | None = None,
    ) -> dict:
        """Lightweight classification only — no retrieval, no LLM generation.

        Returns {needs_clarification: bool, clarifying_questions: [...]}
        so the frontend can decide whether to show the clarifying UI
        before committing to a full RAG pipeline call.
        """
        normalized_query = query.strip()
        history = conversation_history or []
        chat_state = (
            postgres_manager.get_conversation_state(chat_id, user_id=user_id)
            if chat_id
            else None
        )
        provider_used = provider or get_settings().default_llm_provider

        classification = await self._classify_query(
            normalized_query, history, provider_used
        )
        classification = self._hydrate_follow_up_context(
            normalized_query, history, classification, chat_state
        )
        classification = _stabilize_follow_up_classification(
            normalized_query, history, classification, chat_state
        )

        if (
            not classification.needs_clarification
            and not classification.clarifying_questions
        ):
            if _needs_clarification_from_classification(
                classification,
                classification.semantic_confidence,
                has_follow_up_context=_has_follow_up_context(
                    history, classification, chat_state
                ),
            ):
                classification.needs_clarification = True
                classification.clarifying_questions = _default_clarifying_questions(
                    normalized_query, classification, history, chat_state
                )

        return {
            "needs_clarification": classification.needs_clarification,
            "clarifying_questions": classification.clarifying_questions or [],
            "clarifying_message": _clarifying_message(
                normalized_query, history, classification
            )
            if classification.needs_clarification
            else "",
            "main_branch": classification.main_branch,
            "audience": classification.audience,
        }

    async def answer_query_stream_safe(
        self,
        query: str,
        provider: str | None = None,
        conversation_history: list[str] | None = None,
        chat_id: str | None = None,
        active_document_id: str | None = None,
        user_id: str | None = None,
    ):
        """Stream-safe wrapper: checks for vague queries BEFORE retrieval/LLM."""
        import json as _json
        from app.core.logger import get_logger

        log = get_logger(__name__)

        preflight = await self.preflight_classify(
            query, provider, conversation_history, chat_id, user_id
        )
        if preflight.get("needs_clarification") and not active_document_id:
            log.info("query is vague, returning clarifying mode: %s", query[:80])
            yield (
                "data: "
                + _json.dumps(
                    {
                        "answer_mode": "clarifying",
                        "clarifying_questions": preflight["clarifying_questions"],
                        "answer": preflight.get("clarifying_message", ""),
                        "done": True,
                    }
                )
                + "\n\n"
            )
            return

        async for chunk in self.answer_query_stream(
            query,
            provider=provider,
            conversation_history=conversation_history,
            chat_id=chat_id,
            active_document_id=active_document_id,
            user_id=user_id,
        ):
            yield chunk

    async def answer_query_stream(
        self,
        query: str,
        provider: str | None = None,
        conversation_history: list[str] | None = None,
        chat_id: str | None = None,
        active_document_id: str | None = None,
        user_id: str | int | None = None,
    ):
        """Stream answer tokens via SSE, then yields a final `done: true` event.

        Reuses the same classification / retrieval / validation pipeline as
        ``answer_query`` but streams tokens from the LLM so the frontend can
        render progressive output.
        """
        import json as _json

        normalized_query = (query or "").strip()
        history = conversation_history or []
        current_chat_id = chat_id
        provider_used = provider or get_settings().default_llm_provider
        chat_state = (
            postgres_manager.get_conversation_state(current_chat_id, user_id=user_id)
            if current_chat_id
            else None
        )

        # --- Phase 1: classification + follow-up hydration ---
        _t0 = _time.perf_counter()
        classification = await self._classify_query(normalized_query, history, provider)
        classification = self._hydrate_follow_up_context(
            normalized_query, history, classification, chat_state
        )
        classification = _stabilize_follow_up_classification(
            normalized_query, history, classification, chat_state
        )
        _t1 = _time.perf_counter()

        # Clarification gate — skip when user has an active document loaded
        if active_document_id:
            classification.needs_clarification = False
        if (
            not classification.needs_clarification
            and not classification.clarifying_questions
        ):
            if _needs_clarification_from_classification(
                classification,
                classification.semantic_confidence,
                has_follow_up_context=_has_follow_up_context(
                    history, classification, chat_state
                ),
            ):
                classification.needs_clarification = True
                classification.clarifying_questions = _default_clarifying_questions(
                    normalized_query, classification, history, chat_state
                )

        if classification.needs_clarification and classification.clarifying_questions:
            clarifying_answer = _clarifying_message(
                normalized_query, history, classification
            )
            if not current_chat_id:
                current_chat_id = postgres_manager.create_chat(
                    title=normalized_query,
                    active_document_id=active_document_id,
                    user_id=user_id,
                )
            postgres_manager.save_query(
                question=normalized_query, answer=clarifying_answer
            )
            yield (
                "data: "
                + _json.dumps(
                    {
                        "answer_mode": "clarifying",
                        "clarifying_questions": classification.clarifying_questions,
                        "answer": clarifying_answer,
                        "done": True,
                    }
                )
                + "\n\n"
            )
            return

        search_query = self._derive_search_query(
            normalized_query, history, classification
        )
        classification.query_text = search_query
        # When a document is active, summarize/simplify is a normal query —
        # the LLM will summarise based on the document context naturally.
        if active_document_id and classification.is_transformation:
            classification = classification.model_copy(
                update={"is_transformation": False}
            )
        is_transformation = classification.is_transformation

        # --- Phase 2: retrieval ---
        _t_retrieval_start = _time.perf_counter()
        if is_transformation:
            retrieval = RetrievalResult(
                classification=classification,
                official_evidence=[],
                user_evidence=[],
                missing_branches=[],
            )
        else:
            expanded_queries = query_expander.expand(search_query, classification)
            results = await asyncio.gather(
                *[
                    legal_retrieval_service.retrieve(
                        qv,
                        classification,
                        conversation_history=history,
                        active_document_id=active_document_id,
                    )
                    for qv in expanded_queries[:1]
                ]
            )
            all_evidences: list = []
            all_user_evidences: list = []
            for partial in results:
                all_evidences.extend(partial.official_evidence)
                all_user_evidences.extend(partial.user_evidence)
            if all_evidences:
                all_evidences = sorted(
                    all_evidences, key=lambda e: e.score, reverse=True
                )[:12]
            combined_evidences = sorted(
                all_evidences + all_user_evidences, key=lambda e: e.score, reverse=True
            )[:12]
            if combined_evidences:
                retrieval = RetrievalResult(
                    classification=classification,
                    official_evidence=all_evidences,
                    user_evidence=all_user_evidences,
                    missing_branches=[],
                    retrieved_chunks=[e.chunk for e in combined_evidences],
                )
            else:
                retrieval = await legal_retrieval_service.retrieve(
                    classification.search_query or normalized_query,
                    classification,
                    conversation_history=history,
                    active_document_id=active_document_id,
                )
            retrieval = self._filter_retrieval_by_branch(retrieval, classification)
        _t_retrieval_done = _time.perf_counter()

        # Safety net — force user doc chunks into context when a document is active
        try:
            with open("/tmp/pipeline_debug.log", "a") as f:
                f.write(
                    f"[safety_net] active_doc={'YES' if active_document_id else 'NO'} user_ev_count={len(retrieval.user_evidence)} official_ev_count={len(retrieval.official_evidence)} chunks_count={len(retrieval.retrieved_chunks)}\n"
                )
                for ue in retrieval.user_evidence[:3]:
                    f.write(
                        f"  ue: bucket={ue.source_bucket} title={ue.chunk.title} src={ue.chunk.source_scope} score={ue.score:.1f}\n"
                    )
        except Exception:
            pass
        if active_document_id and not retrieval.user_evidence:
            logger.info(
                "User document not found in retrieval — force-fetching chunks for %s",
                active_document_id[:8],
            )
            try:
                from app.services.rag.retriever import retriever_service

                force_chunks = await retriever_service.retrieve(
                    normalized_query or "",
                    where={"document_id": active_document_id},
                )
                if force_chunks:
                    from app.services.legal.retrieval import _source_bucket

                    evs = []
                    for ch in force_chunks:
                        evs.append(
                            type(
                                "Ev",
                                (),
                                {
                                    "chunk": ch,
                                    "score": 30.0,
                                    "source_bucket": _source_bucket(ch),
                                    "retrieval_reason": "force_user_doc",
                                },
                            )()
                        )
                    retrieval = replace(retrieval, user_evidence=evs)
                    retrieval.retrieved_chunks = [
                        e.chunk for e in evs
                    ] + retrieval.retrieved_chunks
                    logger.info(
                        "Force-added %d user doc chunks to context", len(force_chunks)
                    )
            except Exception as exc:
                logger.warning("Force-fetch user doc chunks failed: %s", exc)

        if not retrieval.retrieved_chunks and not is_transformation:
            answer_text = "Não encontrei contexto jurídico suficiente no índice actual para responder com segurança. Reformule a pergunta com mais detalhe, indique o diploma pretendido ou peça o artigo exacto a confirmar."
            if not current_chat_id:
                current_chat_id = postgres_manager.create_chat(
                    title=normalized_query,
                    active_document_id=active_document_id,
                    user_id=user_id,
                )
            yield (
                "data: "
                + _json.dumps(
                    {
                        "done": True,
                        "answer": answer_text,
                        "provider_used": provider_used,
                        "chat_id": current_chat_id,
                        "answer_mode": "refused",
                        "sources": [],
                        "validation_issues": [
                            {
                                "code": "no_context",
                                "message": "Sem contexto jurídico suficiente no índice actual.",
                                "severity": "high",
                            }
                        ],
                    }
                )
                + "\n\n"
            )
            return

        prompt = legal_composer.build_prompt(
            normalized_query,
            classification,
            retrieval.retrieved_chunks,
            conversation_history=history,
        )

        # --- Phase 3: LLM streaming ---
        _t_llm_start = _time.perf_counter()
        accumulated: list[str] = []
        try:
            async for token in self._stream_llm(
                prompt, provider, classification.audience
            ):
                accumulated.append(token)
                yield f"data: {_json.dumps({'token': token})}\n\n"
        except RuntimeError as exc:
            err_text = str(exc)
            yield (
                "data: "
                + _json.dumps(
                    {"done": True, "answer": err_text, "answer_mode": "refused"}
                )
                + "\n\n"
            )
            return

        raw_answer = "".join(accumulated)
        _t_llm_done = _time.perf_counter()

        # --- Phase 4: validate, compose, persist ---
        _t_post_start = _time.perf_counter()
        answer_draft = legal_composer.parse_llm_json(raw_answer)
        answer_draft = legal_composer.constrain_draft_to_context(
            answer_draft, retrieval.retrieved_chunks
        )
        validation = legal_validation_service.validate(
            classification, retrieval, answer_draft
        )
        if is_transformation and answer_draft.rich_content:
            validation.sufficient_legal_support = True
            validation.answer_mode = "grounded"
            validation.issues = []
        confidence = legal_confidence_service.score(
            classification, retrieval, validation, []
        )
        sources = self._select_sources(retrieval, validation)
        if active_document_id and not any(
            s.source_scope == "user_upload" for s in sources
        ):
            logger.info("Force-adding user doc sources for %s", active_document_id[:8])
            try:
                for ev in retrieval.user_evidence:
                    chunk = ev.chunk
                    sources.append(
                        SourceItem(
                            title=chunk.title,
                            source=chunk.source,
                            link_original=chunk.link_original,
                            deep_link=(
                                f"{chunk.link_original}#page={chunk.page}"
                                if chunk.link_original
                                and chunk.page
                                and "#page=" not in chunk.link_original
                                else chunk.link_original
                            ),
                            page=chunk.page,
                            article_number=chunk.article_number,
                            law_status=chunk.law_status,
                            excerpt=chunk.text[:780],
                            attribution_text=chunk.text[:300] if chunk.text else None,
                            source_scope=chunk.source_scope,
                            document_id=chunk.document_id,
                        )
                    )
            except Exception as exc:
                logger.warning("Force-add sources failed: %s", exc)
        answer = legal_composer.compose_answer(
            classification, answer_draft, validation, confidence, sources
        )
        answer = legal_composer.sanitize_answer(answer)
        answer = _normalize_brackets(answer)

        if not current_chat_id:
            current_chat_id = postgres_manager.create_chat(
                title=normalized_query,
                active_document_id=active_document_id,
                user_id=user_id,
            )
        postgres_manager.append_chat_exchange(
            chat_id=current_chat_id,
            question=normalized_query,
            answer=answer,
            provider_used=provider_used or "deepseek",
            sources=[s.model_dump() for s in sources],
            active_document_id=active_document_id,
        )
        postgres_manager.save_query(question=normalized_query, answer=answer)

        postgres_manager.upsert_conversation_state(
            chat_id=current_chat_id,
            user_id=user_id,
            topic_route=classification.topic_route,
            legal_branch=classification.main_branch,
            diploma_slug=(
                self._basis_slug_from_retrieval(
                    validation.confirmed_legal_basis[0], retrieval
                )
                if validation.confirmed_legal_basis
                else None
            ),
            metadata={
                "last_requested_article": (
                    classification.requested_article_numbers[0]
                    if classification.requested_article_numbers
                    else None
                ),
                "last_requested_diploma": (
                    classification.requested_diplomas[0]
                    if classification.requested_diplomas
                    else None
                ),
                "last_answer_mode": validation.answer_mode,
                "last_issue_codes": [i.code for i in validation.issues],
                "normative_status": validation.normative_status,
            },
        )

        _t_post_done = _time.perf_counter()

        # Timing summary
        t_classify = _t1 - _t0
        t_retrieval = _t_retrieval_done - _t_retrieval_start
        t_llm = _t_llm_done - _t_llm_start
        t_post = _t_post_done - _t_post_start
        t_total = _t_post_done - _t0
        logger.info(
            "RAG timing — classify: %(c).1fs | retrieve: %(r).1fs | llm: %(l).1fs | post: %(p).1fs | TOTAL: %(t).1fs",
            {"c": t_classify, "r": t_retrieval, "l": t_llm, "p": t_post, "t": t_total},
        )

        yield (
            "data: "
            + _json.dumps(
                {
                    "done": True,
                    "answer": answer,
                    "provider_used": provider_used or "deepseek",
                    "chat_id": current_chat_id,
                    "active_document_id": active_document_id,
                    "answer_mode": validation.answer_mode,
                    "confidence": confidence.model_dump(),
                    "classification": classification.model_dump(),
                    "sources": [s.model_dump() for s in sources],
                    "validation_issues": [i.model_dump() for i in validation.issues],
                    "legal_basis": [
                        item.model_dump()
                        for item in validation.confirmed_legal_basis
                        + validation.prudential_legal_basis
                    ],
                    "timing": {
                        "classify": round(t_classify, 2),
                        "retrieve": round(t_retrieval, 2),
                        "llm": round(t_llm, 2),
                        "post": round(t_post, 2),
                        "total": round(t_total, 2),
                    },
                }
            )
            + "\n\n"
        )

    async def _stream_llm(self, prompt: str, provider: str | None, audience: str):
        """Yield tokens from the LLM provider.

        Only DeepSeek supports true token streaming in this codebase.
        All other providers fall back to a non-streaming call and yield
        the full response as a single chunk.
        """
        selected = (
            provider or get_settings().default_llm_provider or "deepseek"
        ).lower()
        max_tokens = 800 if audience == "leigo" else 1200

        if selected == "deepseek":
            async for token in deepseek_client.generate_stream(
                prompt, json_mode=False, max_tokens=max_tokens
            ):
                yield token
            return

        content, _ = await llm_router.generate(
            prompt, provider=provider, json_mode=False, max_tokens=max_tokens
        )
        yield content

    @staticmethod
    def _filter_retrieval_by_branch(
        retrieval: RetrievalResult, classification
    ) -> RetrievalResult:
        branch = classification.main_branch
        if branch not in {
            "penal",
            "laboral",
            "civil",
            "familia",
            "tributario",
            "comercial",
            "constitucional",
            "administrativo",
            "propriedade",
        }:
            return retrieval

        def _chunk_branch_name(chunk) -> str | None:
            return (chunk.metadata or {}).get("legal_branch")

        official_on_branch = [
            ev
            for ev in retrieval.official_evidence
            if _chunk_branch_name(ev.chunk) == branch
        ]
        if not official_on_branch:
            return retrieval

        official_off_branch = [
            ev
            for ev in retrieval.official_evidence
            if _chunk_branch_name(ev.chunk) != branch
        ]

        from dataclasses import replace

        filtered = replace(retrieval)
        filtered.official_evidence = official_on_branch + [
            ev for ev in official_off_branch if ev.score > 3.0
        ]
        chunks = []
        seen_ids = set()
        for ev in filtered.official_evidence + retrieval.user_evidence:
            cid = id(ev.chunk)
            if cid not in seen_ids:
                seen_ids.add(cid)
                chunks.append(ev.chunk)
        filtered.retrieved_chunks = chunks
        return filtered

    @staticmethod
    def _select_sources(retrieval, validation) -> list[SourceItem]:
        selected: list[SourceItem] = []
        seen: set[tuple[str, int | None, str]] = set()
        preferred_articles = {
            str(item.article).replace(".", "")
            for item in validation.confirmed_legal_basis
            + validation.prudential_legal_basis
            if item.article
        }
        preferred_keys = {
            (item.diploma, item.page, item.source_scope)
            for item in validation.confirmed_legal_basis
            + validation.prudential_legal_basis
        }
        ordered_evidence = sorted(
            retrieval.official_evidence + retrieval.user_evidence,
            key=lambda evidence: (
                0
                if (
                    evidence.chunk.title,
                    evidence.chunk.page,
                    evidence.chunk.source_scope,
                )
                in preferred_keys
                else 1,
                0
                if RAGPipeline._chunk_matches_preferred_article(
                    evidence.chunk, preferred_articles
                )
                else 1,
                -evidence.score,
            ),
        )
        for evidence in ordered_evidence:
            chunk = evidence.chunk
            key = (chunk.title, chunk.page, chunk.source_scope, chunk.article_number)
            if key in seen:
                continue
            seen.add(key)
            meta = chunk.metadata or {}
            selected.append(
                SourceItem(
                    title=chunk.title,
                    source=chunk.source,
                    link_original=chunk.link_original,
                    deep_link=(
                        f"{chunk.link_original}#page={chunk.page}"
                        if chunk.link_original
                        and chunk.page
                        and "#page=" not in chunk.link_original
                        else chunk.link_original
                    ),
                    page=chunk.page,
                    article_number=chunk.article_number,
                    law_status=chunk.law_status,
                    excerpt=chunk.text[:780],
                    attribution_text=chunk.text[:300] if chunk.text else None,
                    source_scope=chunk.source_scope,
                    source_kind=meta.get("document_kind"),
                    document_id=chunk.document_id,
                )
            )
            if len(selected) >= 8:
                break
        return selected

    @staticmethod
    def _chunk_matches_preferred_article(chunk, preferred_articles: set[str]) -> bool:
        if not preferred_articles:
            return False
        metadata = chunk.metadata or {}
        refs = [
            str(item).replace(".", "")
            for item in (metadata.get("article_references") or [])
        ]
        if chunk.article_number:
            refs.extend(
                part.strip().replace(".", "")
                for part in chunk.article_number.split(",")
                if part.strip()
            )
        main = metadata.get("article_main")
        if main:
            refs.append(str(main).replace(".", ""))
        return any(ref in preferred_articles for ref in refs)


rag_pipeline = RAGPipeline()
