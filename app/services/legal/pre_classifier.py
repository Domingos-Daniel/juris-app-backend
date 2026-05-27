from __future__ import annotations

import asyncio
import re

from app.services.legal.models import LegalClassification


# ---------------------------------------------------------------------------
# Mapeamento determinístico: padrão regex → campos de classificação seguros
# ---------------------------------------------------------------------------

DIPLOMA_PATTERNS: list[tuple[str, dict]] = [
    # Sucessões / Herança
    (
        r"heran[çc]a|herdeiro|testamento|partilha\s+(de\s+)?bens|sucess[aã]o\s+leg[ií]tima|invent[aá]rio\s+obrigat[oó]rio|falec(ido|eu|imento)|deixou\s+(uma|um)\s+(casa|terreno|bem)",
        {
            "main_branch": "civil",
            "topic_route": "sucessoes",
            "requested_diplomas": ["Código Civil"],
        },
    ),
    # Sociedades Comerciais (antes de "comercial" genérico)
    (
        r"sociedade[s]?\s+(comercia|an.nima|por\s+quota|unipessoal)|lei\s+das\s+sociedades\s+comerciais|lsc\b",
        {
            "main_branch": "comercial",
            "topic_route": "sociedades",
            "requested_diplomas": ["Lei das Sociedades Comerciais"],
        },
    ),
    # Família / Divórcio / Separação / Bens do casal
    (
        r"c[oó]digo\s+(da\s+)?fam[ií].lia|div[oó]rcio|divorciar|separa[çc][aã].o\s+(de\s+)?bens|casamento|guarda\s+(de\s+)?(filhos|menor)|filhos?\s+menores|crian[çc]as?\s+(ficam|ficar)|pens[aã].o\s+de\s+alimentos|filia[çc][aã]o|regime\s+de\s+bens|separei",
        {
            "main_branch": "familia",
            "topic_route": "familia",
            "requested_diplomas": ["Código da Família"],
        },
    ),
    # Laboral
    (
        r"despedimento|despedid[oa]|contrato\s+de\s+trabalho|lei\s+geral\s+do\s+trabalho|reintegra[çc][aã]o|subsidio\s+de\s+férias|sal[aá]rio\s+m[ií]nimo|horas\s+extra|trabalh(ador|adora|o\s+ileg|o\s+formal)|rescis[aã]o\s+de\s+contrato|justa\s+causa|indemniza[çc][aã]o\s+(por|de)\s+despedimento",
        {
            "main_branch": "laboral",
            "topic_route": "laboral",
            "requested_diplomas": ["Lei Geral do Trabalho"],
        },
    ),
    # Tributário (IVA primeiro)
    (
        r"\biva\b|imposto\s+sobre\s+o\s+valor\s+acrescentado|factura[çc][aã]o\s+electr[oó]nica",
        {
            "main_branch": "tributario",
            "topic_route": "iva",
            "requested_diplomas": ["Código Geral Tributário"],
        },
    ),
    # Tributário (geral)
    (
        r"c[oó]digo\s+geral\s+tribut[aá]rio|cgt\b|obriga[çc][aã]o\s+tribut[aá]ria|infracc?[aã]o\s+fiscal|administra[çc][aã]o\s+tribut[aá]ria|imposto\s+(sobre|industrial|predial|de\s+rendimento)",
        {
            "main_branch": "tributario",
            "topic_route": "tributario",
            "requested_diplomas": ["Código Geral Tributário"],
        },
    ),
    # Processo Penal
    (
        r"pris[aã]o\s+preventiva|mandado\s+de\s+(busca|deten[çc][aã]o)|coacc?[aã]o\s+processual|recurso\s+penal|liberdade\s+provis[oó]ria|c[oó]digo\s+(do\s+)?processo\s+penal|cpp\b",
        {
            "main_branch": "penal",
            "topic_route": "cpp",
            "requested_diplomas": ["Código do Processo Penal"],
        },
    ),
    # Penal substantivo
    (
        r"c[oó]digo\s+penal|crime\s+de|tipicidade|dolo|culpa\s+penal|pena\s+de\s+pris[aã]o|burla|furto|homic[ií]dio|corrup[çc][aã]o|v[ií]olencia\s+dom[eé]stica",
        {
            "main_branch": "penal",
            "topic_route": "penal_substantivo",
            "requested_diplomas": ["Código Penal"],
        },
    ),
    # Bilhete de Identidade
    (
        r"bilhete\s+de\s+identidade|\bbia?\b|segunda\s+via\s+(do\s+)?bi|conservat[oó]ria",
        {
            "main_branch": "administrativo",
            "topic_route": "identificacao_civil",
            "requested_diplomas": ["Lei do Bilhete de Identidade"],
        },
    ),
    # Terras / Propriedade
    (
        r"lei\s+de\s+terras|concess[aã]o\s+de\s+terra|terreno\s+(r[uú]stico|urbano)|terreno\b|posse\s+de\s+terra|registo\s+(pred|de\s+terra)|propriedade\s+r[uú]stica|despejad[oa]\b|usucapi[aã]o|legalizar\s+terreno",
        {
            "main_branch": "propriedade",
            "topic_route": "terras",
            "requested_diplomas": ["Lei de Terras"],
        },
    ),
    # Contencioso Administrativo
    (
        r"contencioso\s+administrativo|impugna[çc][aã]o\s+administrativa|recurso\s+contencioso|acto\s+administrativo\s+il[eé]gal",
        {
            "main_branch": "administrativo",
            "topic_route": "contencioso_admin",
            "requested_diplomas": ["Código de Processo do Contencioso Administrativo"],
        },
    ),
    # Constitucional
    (
        r"constitui[çc][aã]o\s+(da\s+rep[uú]blica|angolana)|direito[s]?\s+fundamental|garantia\s+constitucional|fiscaliza[çc][aã]o\s+constitucional|direitos\s+garantidos\s+pela\s+constitui[cç]",
        {
            "main_branch": "constitucional",
            "topic_route": "constitucional",
            "requested_diplomas": ["Constituição da República de Angola"],
        },
    ),
    # Civil / Obrigações (genérico — deve vir por último entre os civis)
    (
        r"c[oó]digo\s+civil|obriga[çc][oõ]es\s+civis|responsabilidade\s+civil|m[uú]tuo\s+(banc[aá]rio)?|hipoteca|penhor|arrendamento\s+civil",
        {
            "main_branch": "civil",
            "topic_route": "civil_obrigacoes",
            "requested_diplomas": ["Código Civil"],
        },
    ),
]

# Padrões de audiência — técnico vs leigo
_TECNICO_PATTERNS = re.compile(
    r"nos\s+termos\s+d[ao]|no\s+[âa]mbito\s+d[ao]|requisitos\s+(legais|formais)|"
    r"enquadramento\s+jur[ií]dico|fundamenta[çc][aã]o\s+legal|artigo\s+\d|"
    r"interpreta[çc][aã]o\s+(restritiva|extensiva)|nulidade|anulab|invalidade|"
    r"prescri[çc][aã]o|cadu[çc]|(advogad|jurista|magistrad|juiz|tribunal)",
    re.IGNORECASE,
)
_LEIGO_PATTERNS = re.compile(
    r"o\s+que\s+fa[çc]o|me\s+ajud[ae]|preciso\s+saber|tenho\s+direito|posso\s+fazer|"
    r"como\s+(fa[çc]o|funciona|posso)|n[aã]o\s+entendo|[eé]\s+crime|podem\s+me",
    re.IGNORECASE,
)


TRANSFORMATION_PATTERNS: list[tuple[str, dict]] = [
    (
        r"resum[aei]|em\s+termos\s+simples|explica\s+como\s+se\s+eu\s+fosse|simplific|fala?\s+como\s+leigo|traduz\s+para\s+leigo|para\s+leigo|simpl[ei]s",
        {
            "is_transformation": True,
            "transformation_type": "simplify",
        },
    ),
    (
        r"fale\s+mais|mais\s+detalhes|detalh[aei]|expliq?u[ei]\s+melhor|continua|continue|aprofund",
        {
            "is_transformation": True,
            "transformation_type": "summarize",
        },
    ),
    (
        r"ent[aã]o\s+?.*resum|faz\s+um\s+resumo|com\s+poucas\s+palavras|d[ií]z\s+s[oó]\s+o\s+essencial|breve\s+resumo",
        {
            "is_transformation": True,
            "transformation_type": "summarize",
        },
    ),
]


def pre_classify(question: str) -> dict:
    """
    Pré-classificador determinístico por regex.

    Retorna um dicionário com overrides de classificação quando detectar
    padrões claros. Dicionário vazio = deixar o LLM classificar livremente.

    Esta função é ZERO-COST (sem chamadas LLM) e não falha.
    """
    q = question.strip()
    overrides: dict = {}

    # 0. Detectar transformações (resumir, simplificar, fale mais)
    for pattern, fields in TRANSFORMATION_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            overrides.update(fields)
            break

    # 1. Detectar diploma/ramo por padrão de regex
    for pattern, fields in DIPLOMA_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            overrides.update(fields)
            break  # Primeiro match ganha (ordenados por especificidade)

    # 2. Detectar audiência
    if re.search(_TECNICO_PATTERNS, q):
        overrides["audience"] = "tecnico"
    elif re.search(_LEIGO_PATTERNS, q):
        overrides["audience"] = "leigo"

    # 3. Se a pergunta contiver um diploma explícito com "artigo X", marcar como técnica
    if re.search(r"artigo\s+\d+", q, re.IGNORECASE):
        overrides.setdefault("audience", "tecnico")
        overrides["requires_strict_corpus_match"] = True

    return overrides


def apply_pre_classification(classification_data: dict, question: str) -> dict:
    """
    Aplica os overrides determinísticos sobre os dados do LLM.

    Regra: o pré-classificador prevalece para main_branch, topic_route
    e requested_diplomas quando o LLM retornou 'indeterminado' ou lista vazia.
    Para audiência, o regex prevalece sempre (é mais confiável que o LLM aqui).
    """
    overrides = pre_classify(question)
    if not overrides:
        return classification_data

    merged = dict(classification_data)

    # main_branch: override se o LLM falhou (retornou indeterminado)
    if "main_branch" in overrides:
        if merged.get("main_branch") in ("indeterminado", None, ""):
            merged["main_branch"] = overrides["main_branch"]

    # topic_route: override se o LLM retornou 'geral'
    if "topic_route" in overrides:
        if merged.get("topic_route") in ("geral", None, ""):
            merged["topic_route"] = overrides["topic_route"]

    # requested_diplomas: enriquecer se estiver vazio
    if "requested_diplomas" in overrides:
        if not merged.get("requested_diplomas"):
            merged["requested_diplomas"] = overrides["requested_diplomas"]

    # audience: o regex prevalece sempre (mais confiável)
    if "audience" in overrides:
        merged["audience"] = overrides["audience"]

    # requires_strict_corpus_match: só activa, nunca desactiva
    if overrides.get("requires_strict_corpus_match"):
        merged["requires_strict_corpus_match"] = True

    # is_transformation / transformation_type: override always (deterministic)
    if overrides.get("is_transformation"):
        merged["is_transformation"] = True
        merged["transformation_type"] = overrides.get(
            "transformation_type", "summarize"
        )

    return merged
