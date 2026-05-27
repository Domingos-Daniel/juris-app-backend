from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from app.db.models import SourceItem
from app.services.legal.models import (
    ConfidenceResult,
    LegalClassification,
    LLMAnswerDraft,
    ValidationResult,
)


SECTION_HEADERS = {
    "leigo": {
        "simple": "Em termos simples",
        "steps": "Passos praticos",
        "distinctions": "Distincoes importantes",
        "legal": "Base legal de apoio",
        "prudence": "Nota prudencial",
        "confidence": "Confianca da resposta",
    },
    "misto": {
        "simple": "Explicacao clara",
        "steps": "Passos praticos",
        "distinctions": "Distincoes e nuances",
        "legal": "Base legal de apoio",
        "prudence": "Nota prudencial",
        "confidence": "Confianca da resposta",
    },
    "tecnico": {
        "simple": "Sintese",
        "steps": "Actuacao pratica",
        "distinctions": "Distincoes tecnicas",
        "legal": "Base legal confirmada",
        "prudence": "Limites e validacao adicional",
        "confidence": "Confianca da resposta",
    },
}


def _clean(text: str) -> str:
    updated = (text or "").strip()
    updated = re.sub(r"\n{3,}", "\n\n", updated)
    return updated


def _source_line(source: SourceItem) -> str:
    article = f", art. {source.article_number}" if source.article_number else ""
    page = f", pag. {source.page}" if source.page else ""
    scope_tag = (
        " (Documento do Utilizador)" if source.source_scope == "user_upload" else ""
    )
    return f"- {source.title}{article}{page}{scope_tag}"


def _article_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    return [
        token.strip().replace(".", "")
        for token in re.split(r"[,;/]", value)
        if token.strip()
    ]


def _context_articles(retrieved_chunks: Iterable) -> set[str]:
    articles: set[str] = set()
    for chunk in retrieved_chunks:
        metadata = chunk.metadata or {}
        article_main = metadata.get("article_main")
        if article_main:
            articles.update(_article_tokens(str(article_main)))
        refs = metadata.get("article_references") or []
        for item in refs:
            articles.update(_article_tokens(str(item)))
        if chunk.article_number:
            articles.update(_article_tokens(chunk.article_number))
        for match in re.finditer(
            r"(?:art|artigo|artigos)\s*(\d+[.]?\d*)", chunk.text or "", re.IGNORECASE
        ):
            articles.add(match.group(1).replace(".", ""))
    return {item for item in articles if item}


def _normalized_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"\w+", text or "", re.UNICODE)
        if len(token) >= 4
    }


def _question_anchor_tokens(question: str) -> set[str]:
    stopwords = {
        "como",
        "qual",
        "quais",
        "para",
        "sobre",
        "entre",
        "agora",
        "mesmo",
        "caso",
        "explica",
        "explique",
        "forma",
        "quero",
        "posso",
        "devo",
        "seria",
        "isso",
        "isto",
        "dessa",
        "desse",
    }
    return {token for token in _normalized_tokens(question) if token not in stopwords}


class LegalComposer:
    def build_prompt(
        self,
        question: str,
        classification: LegalClassification,
        retrieved_chunks: list,
        conversation_history: list[str] | None = None,
    ) -> str:
        context_blocks: list[str] = []
        allowed_articles: list[str] = []
        for chunk in retrieved_chunks[:8]:
            branch = (chunk.metadata or {}).get("legal_branch", "indeterminado")
            source_type = (
                "fonte oficial"
                if chunk.source_scope == "official"
                else "documento do utilizador"
            )
            article_main = (
                (chunk.metadata or {}).get("article_main")
                or chunk.article_number
                or "N/D"
            )
            if article_main and article_main != "N/D":
                allowed_articles.append(
                    str(article_main).split(",")[0].strip().replace(".", "")
                )
            context_blocks.append(
                f"[{source_type} | ramo={branch} | diploma={chunk.title} | pagina={chunk.page or 'N/D'} | artigo_principal={article_main} | artigo_contexto={chunk.article_number or 'N/D'}]\n{chunk.text[:500]}"
            )

        whitelist = sorted({item for item in allowed_articles if item})
        whitelist_text = (
            ", ".join(whitelist)
            if whitelist
            else "nenhum artigo confirmado para citacao especifica"
        )
        history = (
            "\n".join(conversation_history[-6:])
            if conversation_history
            else "Sem historico relevante."
        )

        audience = classification.audience
        audience_guidance = ""
        if audience == "leigo":
            audience_guidance = (
                "A TUA AUDIENCIA E UM CIDADAO COMUM (LEIGO).\n"
                "- Traduz a lei para linguagem do dia-a-dia, sem juridiques complicado.\n"
                "- Se pratico: diz exactamente o que fazer e a quem se dirigir.\n"
                "- **NUNCA uses abertura 'Caro Cidadão' ou 'Prezado' ou 'Caro utilizador'. Variedade natural.**\n"
                "- Se base legal insuficiente, inclui nota de cautela simples.\n"
                "- **TAMANHO PROPORCIONAL A COMPLEXIDADE.** Pergunta simples = 1 paragrafo directo. Pergunta complexa = maximo 3 paragrafos. So alonga se o user pedir explicitamente ('detalhe', 'explique', etc.).\n"
                "- **SE PEDIR 'RESUMO' ou 'RESUMA': condensa a resposta anterior a 1 paragrafo curto com apenas os pontos essenciais, sem repetir exemplos.**\n"
            )
        elif audience == "tecnico":
            audience_guidance = (
                "A TUA AUDIENCIA E UM PROFISSIONAL DO DIREITO (ADVOGADO/JURISTA).\n"
                "- Tom estritamente tecnico, objectivo e rigoroso.\n"
                "- Aborda prazos, requisitos de validade, excepcoes e interpretacao.\n"
                "- Indica meios processuais ou de reaccao legais aplicaveis.\n"
                "- NUNCA uses abertura como 'Caro' ou 'Prezado'. Responde directo ao tema.\n"
                "- TAMANHO PROPORCIONAL: pergunta directa = resposta directa. So detalha se pedido.\n"
                "- SE algum artigo ou diploma nao estiver totalmente confirmado no contexto, menciona essa limitacao tecnicamente.\n"
            )
        else:
            audience_guidance = (
                "A TUA AUDIENCIA E UM PUBLICO MISTO OU ESTUDANTE.\n"
                "- Tom didactico, claro e estruturado.\n"
                "- Combina explicacao facil com distincoes tecnicas necessarias.\n"
                "- NUNCA uses abertura 'Caro Cidadão' ou 'Prezado'. Responde directo.\n"
                "- TAMANHO PROPORCIONAL: simples = breve. So detalha se pedido.\n"
                "- SE a base legal for parcial, inclui nota prudencial adequada.\n"
            )
        follow_up_guidance = ""
        if classification.specificity == "follow_up" and conversation_history:
            follow_up_guidance = (
                "CONTEXTO DE FOLLOW-UP:\n"
                "O utilizador esta a dar seguimento a uma conversa anterior. A pergunta actual e uma continuacao do mesmo tema.\n"
                "NAO mudes de assunto. Aprofunda e complementa a resposta anterior com mais detalhes, exemplos praticos e artigos adicionais do mesmo diploma.\n"
                "Evita repetir exactamente o que ja foi dito no historico. Acrescenta valor novo.\n\n"
            )

        return (
            "Es um advogado angolano senior com mais de 20 anos de pratica, especialista em todos os ramos do Direito Angolano.\n"
            "A tua missao e analisar o Contexto Juridico Recuperado e dar uma resposta juridica profissional, clara e estritamente ancorada nesse contexto.\n"
            "Respondes com rigor e sem inventar normas, artigos, valores, prazos ou procedimentos que nao estejam sustentados no contexto fornecido.\n"
            "Responde APENAS com um unico objecto JSON valido.\n"
            "Usa exactamente as seguintes chaves:\n"
            "  - rich_content: resposta em Markdown. PARA TODOS OS CASOS usa pelo menos um destes elementos de formatacao:\n"
            "    * Listas numeradas (1. X) ou bullet (- X) quando enumeras direitos, condicoes, passos ou requisitos.\n"
            "    * Subtitulos (###) quando a resposta tem duas ou mais seccoes logicas.\n"
            "    * Paragrafos separados por linha em branco entre topicos distintos.\n"
            "  - cited_articles: Lista de artigos citados no texto.\n"
            "  - cited_diplomas: Lista de diplomas citados no texto.\n"
            "\n"
            "REGRAS CRITICAS DE ANALISE JURIDICA:\n"
            "1. USA APENAS O CONTEXTO COMO BASE. Nao completes com conhecimento geral do modelo.\n"
            "2. SE O CONTEXTO NAO CONFIRMAR O PONTO EXACTO, DIZ ISSO EXPRESSAMENTE. Podes explicar o limite do material recuperado, mas nao inventes resposta normativa.\n"
            "3. NUNCA INVENTES artigos, numeros, custos, prazos, orgaos competentes ou formulas do tipo 'em geral'.\n"
            "4. COBERTURA COMPLETA: responde a pergunta com o que o contexto efectivamente permite afirmar. Se houver lacuna, identifica a lacuna.\n"
            "5. CITACOES: Sempre que mencionares um artigo sustentado no contexto, escreve a referencia EXATAMENTE neste formato (COM colchetes): [[Art. X, Diploma Y, p. Z]]. Exemplo: [[Art. 300.o, Lei Geral do Trabalho, p. 155]]. O sistema converte para apresentacao visual.\n"
            "6. SEM REDUNDANCIA: nao facas listas finais de fontes; o sistema mostra isso na interface.\n"
            "7. RECUPERACAO E APOLOGIA: se for uma CORRECCAO, inicia a rich_content com um pedido de desculpas profissional.\n"
            "8. ESTILO: Portugues de Angola, formal, claro e proporcional a complexidade.\n"
            "9. PROIBIDO COLCHETES DUPLOS: usa apenas [ ] simples.\n"
            "10. SO PREENCHES cited_articles com artigos confirmados na whitelist abaixo.\n"
            "\n"
            f"ARTIGOS CONFIRMADOS NO CONTEXTO (WHITELIST): {whitelist_text}\n\n"
            f"{audience_guidance}\n"
            f"{follow_up_guidance}"
            f"TIPO DE RESPOSTA: {'CORRECCAO' if classification.is_correction else 'TRANSFORMACAO' if classification.is_transformation else 'RESPOSTA NORMAL'}\n"
            f"OBJECTIVO: {classification.transformation_type if classification.is_transformation else 'Analise Juridica'}\n"
            f"Ramo: {classification.main_branch} | Audiencia: {classification.audience} | Topico: {classification.topic_route}\n"
            f"Pergunta do utilizador: {question}\n"
            f"Historico resumido: {history}\n\n"
            "Contexto juridico recuperado:\n" + "\n\n---\n\n".join(context_blocks)
        )

    def parse_llm_json(self, raw_answer: str) -> LLMAnswerDraft:
        cleaned_raw = self.sanitize_answer(raw_answer)
        if not cleaned_raw:
            return LLMAnswerDraft()
        payload = None
        try:
            payload = json.loads(cleaned_raw)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", cleaned_raw)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except Exception:
                    payload = None
        if not isinstance(payload, dict):
            return LLMAnswerDraft(rich_content=_clean(cleaned_raw))

        if "json_object" in payload and isinstance(payload["json_object"], dict):
            payload = payload["json_object"]
        if "json" in payload and isinstance(payload["json"], dict):
            payload = payload["json"]
        for wrapper in ("rich_content", "answer", "response"):
            if wrapper in payload and isinstance(payload[wrapper], dict):
                payload = payload[wrapper]
                break

        for list_key in (
            "practical_steps",
            "distinctions",
            "prudent_inferences",
            "additional_validation_needed",
            "cited_articles",
            "cited_diplomas",
        ):
            value = payload.get(list_key)
            if value in (None, "", False, True):
                payload[list_key] = (
                    []
                    if value in (None, "", False)
                    else ["Validacao adicional assinalada pelo modelo."]
                )
            elif isinstance(value, str):
                payload[list_key] = [_clean(value)] if _clean(value) else []
            elif isinstance(value, list):
                payload[list_key] = [
                    _clean(str(item)) for item in value if _clean(str(item))
                ]
            else:
                payload[list_key] = [_clean(str(value))] if _clean(str(value)) else []
        return LLMAnswerDraft(**payload)

    def constrain_draft_to_context(
        self, draft: LLMAnswerDraft, retrieved_chunks: list
    ) -> LLMAnswerDraft:
        context_articles = _context_articles(retrieved_chunks)
        normalized_articles = []
        for article in draft.cited_articles:
            normalized = str(article).strip().replace(".", "")
            if normalized and normalized in context_articles:
                normalized_articles.append(normalized)
        normalized_diplomas = []
        available_titles = {
            _clean(chunk.title).casefold(): chunk.title
            for chunk in retrieved_chunks
            if _clean(chunk.title)
        }
        for diploma in draft.cited_diplomas:
            cleaned = _clean(str(diploma))
            if not cleaned:
                continue
            if cleaned.casefold() in available_titles:
                normalized_diplomas.append(available_titles[cleaned.casefold()])
                continue
            if any(cleaned.casefold() in title for title in available_titles):
                normalized_diplomas.append(cleaned)
        draft.cited_articles = list(dict.fromkeys(normalized_articles))
        draft.cited_diplomas = list(dict.fromkeys(normalized_diplomas))
        return draft

    def sanitize_answer(self, answer: str) -> str:
        updated = answer.strip()
        updated = re.sub(r"^```(?:json)?\s*", "", updated)
        updated = re.sub(r"\s*```$", "", updated)
        updated = re.sub(r"(?:\x1f|\x1e|\x1d|\x1c)", "", updated)
        return updated.strip()

    @staticmethod
    def _extract_rich_content(text: str) -> str:
        if not text:
            return ""
        for key in ("rich_content", "direct_answer", "simple_explanation"):
            pattern = rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return match.group(1).encode().decode("unicode_escape")
                except Exception:
                    pass
        return ""

    def fallback_from_validation(
        self,
        validation: ValidationResult,
        original_draft: LLMAnswerDraft | None = None,
    ) -> LLMAnswerDraft:
        issue_codes = {issue.code for issue in validation.issues}
        if "followup_anchor_unresolved" in issue_codes:
            return LLMAnswerDraft(
                rich_content="Nao confirmei ainda o artigo exacto a que o follow-up se refere. Para responder com seguranca, preciso recuperar e validar primeiro esse mesmo artigo ou o diploma correspondente."
            )
        if "requested_article_not_recovered" in issue_codes:
            return LLMAnswerDraft(
                rich_content="O artigo exacto pedido nao foi confirmado no contexto recuperado. Posso continuar a procurar esse artigo, mas nao devo responder como se ele tivesse sido localizado."
            )
        if "normative_conflict" in issue_codes or "citator_gap" in issue_codes:
            return LLMAnswerDraft(
                rich_content="O contexto recuperado indica possivel conflito, alteracao ou revogacao normativa, mas ainda nao confirma com seguranca qual e o regime juridico prevalecente para esta resposta."
            )
        if "vigency_unverified" in issue_codes:
            return LLMAnswerDraft(
                rich_content="A pergunta exige confirmacao de vigencia normativa, e o contexto actual ainda nao permite afirmar com seguranca se a norma continua em vigor."
            )

        if original_draft and original_draft.rich_content:
            return original_draft

        if validation.answer_mode == "refused":
            rich = "Nao foi possivel encontrar informacao juridica suficiente para responder com seguranca. Tente reformular a pergunta com mais contexto."
        elif validation.answer_mode == "grounded_with_caveat":
            if original_draft and original_draft.rich_content:
                return original_draft
            rich = "A informacao disponivel no momento e parcial. A resposta que se segue baseia-se no contexto recuperado e no conhecimento geral do Direito Angolano."
        else:
            rich = "A informacao disponivel no momento nao permite uma resposta completa. Recomendo reformular com mais detalhes."

        return LLMAnswerDraft(rich_content=rich)

    def answer_tracks_question(
        self, question: str, draft: LLMAnswerDraft, classification: LegalClassification
    ) -> bool:
        question_tokens = _question_anchor_tokens(question)
        if not question_tokens:
            return True
        answer_tokens = _normalized_tokens(draft.rich_content)
        overlap = question_tokens.intersection(answer_tokens)
        return len(overlap) >= 1

    def answer_looks_like_json_artifact(self, answer: str) -> bool:
        stripped = answer.lstrip()
        return stripped.startswith("{") or stripped.startswith("```")

    def compose_answer(
        self,
        classification: LegalClassification,
        draft: LLMAnswerDraft,
        validation: ValidationResult,
        confidence: ConfidenceResult,
        sources: list[SourceItem],
    ) -> str:
        if draft.rich_content:
            return _clean(draft.rich_content)
        return ""


legal_composer = LegalComposer()
