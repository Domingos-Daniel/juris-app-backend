from __future__ import annotations

import hashlib
import json
import gc
import re
import unicodedata
from pathlib import Path
from typing import TypedDict

import requests

from app.core.config import get_settings
from app.core.logger import get_logger
from app.db.models import DocumentIngestResponse, IngestSummary
from app.services.legal.models import (
    DocumentRole,
    LegalBranch,
    NormTypeNeeded,
    TopicRoute,
)
from app.services.pdf.chunker import legal_semantic_chunks
from app.services.pdf.extractor import (
    extract_pages_from_pdf,
    extract_pages_from_pdf_range,
)
from app.services.pdf.ocr import ocr_service
from app.services.rag.vector_store import legislation_vector_store

logger = get_logger(__name__)
ARTICLE_RE = re.compile(
    r"(?:art|artigo|artigos|art\.?º?)\s*(\d+[.]?\d*)", re.IGNORECASE
)
STRUCTURAL_RE = re.compile(
    r"\b(indice|índice|sumario|sumário|capitulo|capítulo|sec[cç][aã]o|titulo|título)\b",
    re.IGNORECASE,
)
NORMATIVE_TERMS = (
    "deve",
    "direito",
    "direitos",
    "obrigação",
    "obrigacao",
    "compete",
    "pena",
    "contrato",
    "trabalhador",
    "empregador",
    "processo",
)
TITLE_STOPWORDS = {
    "a",
    "ao",
    "as",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "lei",
    "no",
    "o",
    "os",
    "para",
    "por",
}
STRICT_TOPIC_ROUTES = {
    "cpp",
    "cpc",
    "contencioso_admin",
    "processo_administrativo",
    "identificacao_civil",
    "estatuto_magistrados",
    "tribunal_supremo_organica",
    "tributario",
    "iva",
    "sociedades",
    "familia",
    "terras",
}


class OfficialDocumentSpec(TypedDict):
    filename: str
    title: str
    url: str
    legal_branch: LegalBranch
    diploma_slug: str
    document_kind: str
    document_role: DocumentRole
    norm_type: NormTypeNeeded
    topic_route: TopicRoute
    primary_topics: list[str]
    exclusive_domains: list[str]
    source_priority: float
    is_primary_source: bool
    law_status: str


OFFICIAL_DOCUMENT_CATALOG: list[OfficialDocumentSpec] = [
    {
        "filename": "Constituicao-da-Republica-2022.pdf",
        "title": "Constituicao da Republica de Angola (2022)",
        "url": "https://www.tribunalconstitucional.ao/media/3nypgra0/edicao-especial-actualizada-2022.pdf",
        "legal_branch": "constitucional",
        "diploma_slug": "constituicao-republica-angola-2022",
        "document_kind": "legislation_official",
        "document_role": "constituicao",
        "norm_type": "substantiva",
        "topic_route": "constitucional",
        "primary_topics": [
            "direitos_fundamentais",
            "fiscalizacao_constitucional",
            "garantias_constitucionais",
        ],
        "exclusive_domains": ["constitucionalidade", "colisao_direitos_fundamentais"],
        "source_priority": 1.0,
        "is_primary_source": True,
        "law_status": "Em vigor",
    },
    {
        "filename": "Codigo-Penal-Lei-38-20.pdf",
        "title": "Codigo Penal (Lei 38/20)",
        "url": "https://faolex.fao.org/docs/pdf/ang199073.pdf",
        "legal_branch": "penal",
        "diploma_slug": "codigo-penal-lei-38-20",
        "document_kind": "legislation_official",
        "document_role": "codigo_base",
        "norm_type": "substantiva",
        "topic_route": "penal_substantivo",
        "primary_topics": ["crimes", "penas", "violencia_domestica", "tipicidade"],
        "exclusive_domains": ["crime", "pena", "tipicidade"],
        "source_priority": 0.98,
        "is_primary_source": True,
        "law_status": "Em vigor",
    },
    {
        "filename": "Codigo-Processo-Penal-Lei-39-20.pdf",
        "title": "Codigo do Processo Penal (Lei 39/20)",
        "url": "https://tribunalsupremo.ao/wp-content/uploads/2023/03/C%C3%B3digo-Penal-e-do-Processo-Penal-Angolanos-2020-DRI-179_11-Novembro-176_230110_151357-1.pdf",
        "legal_branch": "penal",
        "diploma_slug": "codigo-processo-penal-lei-39-20",
        "document_kind": "legislation_official",
        "document_role": "codigo_processual",
        "norm_type": "processual",
        "topic_route": "cpp",
        "primary_topics": [
            "prisao_preventiva",
            "mandado_de_busca",
            "recurso_penal",
            "medidas_de_coaccao",
        ],
        "exclusive_domains": [
            "prazo_processual_penal",
            "rito_penal",
            "prisao",
            "mandado_de_busca",
        ],
        "source_priority": 1.0,
        "is_primary_source": True,
        "law_status": "Em vigor",
    },
    {
        "filename": "Lei-Geral-do-Trabalho-Lei-12-23.pdf",
        "title": "Lei Geral do Trabalho (Lei 12/23)",
        "url": "https://www.lgt.gov.ao/wp-content/uploads/2025/05/Livro-da-Lei-Geral-do-Trabalho.pdf",
        "legal_branch": "laboral",
        "diploma_slug": "lei-geral-do-trabalho-lei-12-23",
        "document_kind": "legislation_official",
        "document_role": "lei_especial",
        "norm_type": "substantiva",
        "topic_route": "laboral",
        "primary_topics": ["despedimento", "reintegracao", "indemnizacao", "salario"],
        "exclusive_domains": ["relacao_laboral", "contrato_de_trabalho"],
        "source_priority": 1.0,
        "is_primary_source": True,
        "law_status": "Em vigor",
    },
    {
        "filename": "Codigo-Civil.pdf",
        "title": "Codigo Civil",
        "url": "https://www.africa-laws.org/Angola/civil%20law/Civil%20Code%20(%20in%20Portuguese).pdf",
        "legal_branch": "civil",
        "diploma_slug": "codigo-civil",
        "document_kind": "legislation_official",
        "document_role": "codigo_base",
        "norm_type": "substantiva",
        "topic_route": "civil_obrigacoes",
        "primary_topics": [
            "mutuo",
            "contratos",
            "obrigacoes",
            "responsabilidade_civil",
        ],
        "exclusive_domains": ["obrigacoes_civis", "mutuo"],
        "source_priority": 0.9,
        "is_primary_source": True,
        "law_status": "Em vigor",
    },
    {
        "filename": "Codigo-Familia-Lei-1-88.pdf",
        "title": "Codigo da Familia (Lei 1/88)",
        "url": "https://files.lex.ao/assembleia-do-povo/1988/lei-n-o-1-88-de-20-de-fevereiro/download/lei-n-o-1-88-de-20-de-fevereiro_assembleia-do-povo_lex-ao.pdf",
        "legal_branch": "familia",
        "diploma_slug": "codigo-familia-lei-1-88",
        "document_kind": "legislation_official",
        "document_role": "codigo_base",
        "norm_type": "substantiva",
        "topic_route": "familia",
        "primary_topics": ["casamento", "divorcio", "filiacao", "sucessoes_familiares"],
        "exclusive_domains": ["direito_da_familia", "regime_de_bens", "filiacao"],
        "source_priority": 1.0,
        "is_primary_source": True,
        "law_status": "Em vigor",
    },
    {
        "filename": "Codigo-Processo-Contencioso-Administrativo-Lei-33-22.pdf",
        "title": "Codigo de Processo do Contencioso Administrativo (Lei 33/22)",
        "url": "https://files.lex.ao/assembleia-nacional/2022/lei-n-o-33-22-de-01-de-setembro/download/lei-n-o-33-22-de-01-de-setembro_assembleia-nacional_lex-ao.pdf",
        "legal_branch": "administrativo",
        "diploma_slug": "codigo-processo-contencioso-administrativo-33-22",
        "document_kind": "legislation_official",
        "document_role": "codigo_processual",
        "norm_type": "processual",
        "topic_route": "contencioso_admin",
        "primary_topics": [
            "impugnacao_administrativa",
            "prazo_contencioso",
            "recurso_contencioso",
        ],
        "exclusive_domains": [
            "contencioso_administrativo",
            "prazo_processual_administrativo",
        ],
        "source_priority": 1.0,
        "is_primary_source": True,
        "law_status": "Em vigor",
    },
    {
        "filename": "Lei-Bilhete-Identidade-Lei-4-09.pdf",
        "title": "Lei do Bilhete de Identidade (Lei 4/09)",
        "url": "https://files.lex.ao/assembleia-nacional/2009/lei-n-o-4-09-de-30-de-junho/download/lei-n-o-4-09-de-30-de-junho_assembleia-nacional_lex-ao.pdf",
        "legal_branch": "administrativo",
        "diploma_slug": "lei-bilhete-identidade-4-09",
        "document_kind": "legislation_official",
        "document_role": "identificacao_civil",
        "norm_type": "administrativo_operacional",
        "topic_route": "identificacao_civil",
        "primary_topics": ["segunda_via_bi", "emissao_bi", "custos_administrativos_bi"],
        "exclusive_domains": ["bilhete_de_identidade", "identificacao_civil"],
        "source_priority": 1.0,
        "is_primary_source": True,
        "law_status": "Em vigor",
    },
    {
        "filename": "Codigo-Geral-Tributario-Lei-21-14.pdf",
        "title": "Codigo Geral Tributario (Lei 21/14)",
        "url": "https://files.lex.ao/assembleia-nacional/2014/lei-n-o-21-14-de-22-de-outubro/download/lei-n-o-21-14-de-22-de-outubro_assembleia-nacional_lex-ao.pdf",
        "legal_branch": "tributario",
        "diploma_slug": "codigo-geral-tributario-21-14",
        "document_kind": "legislation_official",
        "document_role": "tributaria",
        "norm_type": "substantiva",
        "topic_route": "tributario",
        "primary_topics": [
            "impostos",
            "obrigacao_tributaria",
            "facturacao",
            "infraccao_fiscal",
        ],
        "exclusive_domains": ["tributos", "impostos", "codigo_geral_tributario"],
        "source_priority": 0.98,
        "is_primary_source": True,
        "law_status": "Em vigor",
    },
    {
        "filename": "Lei-Terras-Lei-9-04.pdf",
        "title": "Lei de Terras (Lei 9/04)",
        "url": "https://files.lex.ao/assembleia-nacional/2004/lei-n-o-9-04-de-09-de-novembro/download/lei-n-o-9-04-de-09-de-novembro_assembleia-nacional_lex-ao.pdf",
        "legal_branch": "propriedade",
        "diploma_slug": "lei-terras-9-04",
        "document_kind": "legislation_official",
        "document_role": "lei_especial",
        "norm_type": "substantiva",
        "topic_route": "terras",
        "primary_topics": ["concessao", "titulos_de_terra", "uso_e_aproveitamento"],
        "exclusive_domains": ["terrenos", "concessao_de_terra", "lei_de_terras"],
        "source_priority": 0.95,
        "is_primary_source": True,
        "law_status": "Em vigor",
    },
    {
        "filename": "Lei-Sociedades-Comerciais-Lei-1-04.pdf",
        "title": "Lei das Sociedades Comerciais (Lei 1/04)",
        "url": "https://kiwi.fgc.gov.ao/leitor.html?t=Legislacao&r=1006720251002155205101",
        "legal_branch": "comercial",
        "diploma_slug": "lei-sociedades-comerciais-1-04",
        "document_kind": "legislation_official",
        "document_role": "lei_especial",
        "norm_type": "substantiva",
        "topic_route": "sociedades",
        "primary_topics": [
            "socios",
            "quotas",
            "constituicao_de_empresas",
            "tipos_societarios",
        ],
        "exclusive_domains": ["sociedades_comerciais", "quotas", "socios"],
        "source_priority": 0.96,
        "is_primary_source": True,
        "law_status": "Em vigor",
    },
]

BRANCH_BY_TITLE = {
    item["title"]: item["legal_branch"] for item in OFFICIAL_DOCUMENT_CATALOG
}
DIPLOMA_SLUG_BY_TITLE = {
    item["title"]: item["diploma_slug"] for item in OFFICIAL_DOCUMENT_CATALOG
}
SOURCE_PRIORITY_BY_TITLE = {
    item["title"]: item["source_priority"] for item in OFFICIAL_DOCUMENT_CATALOG
}
DOCUMENT_SPEC_BY_TITLE = {item["title"]: item for item in OFFICIAL_DOCUMENT_CATALOG}
OFFICIAL_DOCUMENTS = [
    {"filename": item["filename"], "title": item["title"], "url": item["url"]}
    for item in OFFICIAL_DOCUMENT_CATALOG
]


def _document_spec(title: str) -> OfficialDocumentSpec | None:
    return DOCUMENT_SPEC_BY_TITLE.get(title)


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_only.lower()).strip()


def _expected_signature(title: str) -> tuple[str | None, str | None]:
    match = re.search(r"\(Lei\s+(\d+)\/(\d{2,4})\)", title, re.IGNORECASE)
    if not match:
        return None, None
    return match.group(1).lstrip("0") or "0", match.group(2)[-2:]


def _expected_title_tokens(title: str) -> list[str]:
    base_title = re.sub(r"\s*\(.*\)\s*$", "", title).strip()
    tokens = []
    for token in re.split(r"[^a-z0-9]+", _normalized_text(base_title)):
        if len(token) < 4 or token in TITLE_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _document_signature_matches_text(title: str, text: str) -> bool:
    normalized_text = _normalized_text(text)
    if not normalized_text:
        return False

    number, year = _expected_signature(title)
    if number and year:
        # Padrao mais flexivel usando regex para capturar variacoes de "Lei n.o 1/04"
        # O texto ja esta normalizado (sem acentos, minusculo, espacos simples)
        pattern = (
            rf"lei\s*(?:n\.?\s*o?\s*|no\s+|numero\s+)?\s*{number}\s*/\s*(?:20)?{year}"
        )
        has_signature = bool(re.search(pattern, normalized_text))

        # Backup: apenas o numero e ano proximos se o regex estrito falhar
        if not has_signature:
            has_signature = (
                f"{number}/{year}" in normalized_text
                or f"{number} / {year}" in normalized_text
                or f"{number}/{year[-2:]}" in normalized_text
            )

        if not has_signature:
            logger.debug(
                "Falha na assinatura para '%s': Esperado %s/%s", title, number, year
            )
    else:
        has_signature = True

    expected_tokens = _expected_title_tokens(title)
    matched_tokens = [token for token in expected_tokens if token in normalized_text]

    if len(expected_tokens) <= 3:
        minimum_tokens = max(1, len(expected_tokens) - 1)
    else:
        minimum_tokens = max(
            2, round(len(expected_tokens) * 0.4)
        )  # Mais tolerante (0.4 vs 0.5)

    has_title_identity = len(matched_tokens) >= minimum_tokens

    if not has_title_identity:
        logger.debug(
            "Falha na identidade para '%s': Encontrados %s de %s tokens esperados (%s)",
            title,
            len(matched_tokens),
            len(expected_tokens),
            matched_tokens,
        )

    return has_signature and has_title_identity


def _lex_catalog_pdf_url_for_slug(diploma_slug: str) -> str | None:
    catalog_path = Path("C:/Projectos/TCC/backend/data/catalogs/lex_ao_documents.json")
    if not catalog_path.exists():
        return None
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    for item in payload.get("documents", []):
        if item.get("matched_internal_slug") == diploma_slug and item.get(
            "download_pdf_url"
        ):
            return str(item["download_pdf_url"])
    return None


def _source_url_for_document(document: dict) -> str:
    slug = str(document.get("diploma_slug") or "")
    return _lex_catalog_pdf_url_for_slug(slug) or str(document.get("url") or "")


def _is_pdf_content_usable_for_document(title: str, pdf_path: Path) -> bool:
    try:
        # Otimizacao: Validar apenas as primeiras 4 paginas para checar identidade do documento
        pages, _used_ocr = extract_pages_from_pdf_range(pdf_path, 1, 4)
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.warning("Falha ao validar integridade de %s: %s", pdf_path.name, exc)
        return False

    sampled = "\n".join((page.get("text") or "") for page in pages[:4]).strip()
    if len(sampled) < 120:
        return False
    return _document_signature_matches_text(title, sampled)


def _download_document_to_path(url: str, destination: Path, timeout: int) -> bool:
    response = requests.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    if len(response.content) < 1000:
        return False
    destination.write_bytes(response.content)
    return True


def _resolve_document_path(settings, document: dict) -> Path | None:
    destination = settings.raw_pdfs_dir / document["filename"]
    source_url = _source_url_for_document(document)

    if destination.exists() and destination.stat().st_size > 0:
        if _is_pdf_content_usable_for_document(document["title"], destination):
            return destination
        logger.warning(
            "Ficheiro local rejeitado por integridade semantica: %s", destination.name
        )
        if destination.exists():
            destination.unlink()

    if not source_url.lower().endswith(".pdf"):
        return None

    try:
        logger.info("A baixar %s", source_url)
        ok = _download_document_to_path(
            source_url, destination, settings.request_timeout_seconds
        )
        if not ok:
            logger.warning("Conteudo insuficiente ao preparar %s", document["title"])
            return None
        if not _is_pdf_content_usable_for_document(document["title"], destination):
            logger.warning(
                "PDF descarregado nao corresponde ao diploma esperado: %s",
                document["title"],
            )
            destination.unlink(missing_ok=True)
            return None
        logger.info("Download concluido para %s", document["filename"])
        return destination
    except requests.RequestException as exc:
        logger.warning("Falha ao preparar %s: %s", document["title"], exc)
    except OSError as exc:
        logger.warning("Falha local ao preparar %s: %s", document["title"], exc)
    return None


def _is_structural_page(text: str) -> bool:
    lowered = text.lower()
    return (
        "indice" in lowered
        or "índice" in lowered
        or sum(1 for _ in STRUCTURAL_RE.finditer(lowered)) >= 3
    )


def _looks_like_front_matter(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "preambulo",
            "preâmbulo",
            "texto integral",
            "edição especial",
            "edicao especial",
            "versão original",
            "versao original",
        )
    )


def _extract_article_references(text: str) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for match in ARTICLE_RE.finditer(text or ""):
        article = match.group(1).rstrip(".").replace(".", "")
        if article in seen:
            continue
        seen.add(article)
        items.append(article)
    return items


def _primary_article_number(text: str) -> str | None:
    refs = _extract_article_references(text)
    return refs[0] if refs else None


def _article_reference_summary(text: str, limit: int = 4) -> str | None:
    refs = _extract_article_references(text)
    if not refs:
        return None
    return ", ".join(refs[:limit])


def _normative_density(text: str) -> float:
    lowered = text.lower()
    article_hits = len(_extract_article_references(lowered))
    term_hits = sum(lowered.count(term) for term in NORMATIVE_TERMS)
    return round((article_hits * 1.5) + (term_hits * 0.35), 3)


def _is_normative_chunk(text: str) -> bool:
    return _normative_density(text) >= 1.5


def _document_kind(title: str) -> str:
    spec = _document_spec(title)
    return spec["document_kind"] if spec else "legal_document"


_BRANCH_KEYWORDS: dict[str, list[str]] = {
    "penal": [
        "penal",
        "crime",
        "criminal",
        "pena",
        "prisao",
        "processo penal",
        "contravencao",
    ],
    "civil": [
        "civil",
        "obrigacao",
        "contrato",
        "responsabilidade civil",
        "codigo civil",
    ],
    "laboral": [
        "trabalho",
        "laboral",
        "trabalhador",
        "salario",
        "greve",
        "sindicato",
        "seguranca social",
    ],
    "comercial": [
        "comercial",
        "sociedade",
        "empresa",
        "comercio",
        "insolvencia",
        "falencia",
    ],
    "tributario": [
        "tributario",
        "imposto",
        "fiscal",
        "taxa",
        "contribuicao",
        "iva",
        "aduaneiro",
    ],
    "administrativo": [
        "administrativo",
        "contratacao publica",
        "funcionario publico",
        "contencioso",
        "procedimento administrativo",
        "bilhete",
    ],
    "constitucional": [
        "constituicao",
        "constitucional",
        "direitos fundamentais",
        "organizacao do estado",
    ],
    "familia": [
        "familia",
        "casamento",
        "divorcio",
        "filho",
        "alimento",
        "adopcao",
        "parental",
        "registo civil",
    ],
    "propriedade": [
        "terra",
        "terras",
        "propriedade",
        "expropriacao",
        "urbano",
        "habitacao",
        "ordenamento",
    ],
}
_BRANCH_KEYWORDS_LOWER = {
    branch: [k.lower() for k in keys] for branch, keys in _BRANCH_KEYWORDS.items()
}


def _branch_for_title(title: str) -> str:
    branch = BRANCH_BY_TITLE.get(title)
    if branch and branch != "indeterminado":
        return branch
    title_lower = title.lower()
    scored: list[tuple[int, str]] = []
    for branch_name, keywords in _BRANCH_KEYWORDS_LOWER.items():
        hits = sum(1 for kw in keywords if kw in title_lower)
        if hits:
            scored.append((hits, branch_name))
    if scored:
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]
    return "indeterminado"


def _diploma_slug(title: str) -> str:
    return DIPLOMA_SLUG_BY_TITLE.get(
        title, re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    )


def _source_priority(title: str) -> float:
    return SOURCE_PRIORITY_BY_TITLE.get(title, 0.7)


def _document_role(title: str) -> str:
    spec = _document_spec(title)
    return spec["document_role"] if spec else "lei_especial"


def _norm_type(title: str) -> str:
    spec = _document_spec(title)
    return spec["norm_type"] if spec else "misto"


def _topic_route(title: str) -> str:
    spec = _document_spec(title)
    return spec["topic_route"] if spec else "geral"


def _primary_topics(title: str) -> list[str]:
    spec = _document_spec(title)
    return list(spec["primary_topics"]) if spec else []


def _exclusive_domains(title: str) -> list[str]:
    spec = _document_spec(title)
    return list(spec["exclusive_domains"]) if spec else []


def _routing_priority(title: str) -> float:
    spec = _document_spec(title)
    if not spec:
        return 0.7
    return round(
        float(spec["source_priority"]) + (0.1 if spec["is_primary_source"] else 0.0), 3
    )


def _strict_topics(title: str) -> list[str]:
    route = _topic_route(title)
    return [route] if route in STRICT_TOPIC_ROUTES else []


def _catalog_metadata(title: str) -> dict:
    return {
        "catalog_version": "2026-05-routing-v1",
        "document_role": _document_role(title),
        "norm_type": _norm_type(title),
        "topic_route": _topic_route(title),
        "primary_topics": _primary_topics(title),
        "exclusive_domains": _exclusive_domains(title),
        "strict_topics": _strict_topics(title),
        "routing_priority": _routing_priority(title),
        "is_primary_source": bool(
            _document_spec(title) and _document_spec(title)["is_primary_source"]
        ),
    }


def _page_reference_density(page_text: str) -> int:
    return len(_extract_article_references(page_text))


def _is_context_heavy_page(page_text: str) -> bool:
    return _page_reference_density(page_text) >= 3


def _chunk_priority(article_main: str | None, segmentation: str) -> float:
    score = 1.0
    if article_main:
        score += 1.2
    if segmentation == "article_block":
        score += 1.0
    return score


def _chunk_kind(segmentation: str) -> str:
    if segmentation == "article_block":
        return "article_normative"
    return "text_normative"


def build_official_metadata(
    document: dict,
    page_number: int,
    chunk: str,
    page_used_ocr: bool,
    chunk_index: int,
    law_status: str,
    article_number: str | None,
) -> dict:
    title = document["title"]
    metadata = {
        "source": document["path"].name,
        "title": title,
        "link_original": document["url"],
        "page": page_number,
        "article_number": article_number,
        "law_status": law_status,
        "used_ocr": page_used_ocr,
        "chunk_index": chunk_index,
        "source_scope": "official",
        "document_id": None,
        "legal_branch": _branch_for_title(title),
        "diploma_slug": _diploma_slug(title),
        "is_front_matter": _looks_like_front_matter(chunk),
        "is_structural": _is_structural_page(chunk),
        "normative_density": _normative_density(chunk),
        "is_normative": _is_normative_chunk(chunk),
        "source_priority": max(_source_priority(title), _routing_priority(title)),
        "document_kind": _document_kind(title),
    }
    metadata.update(_catalog_metadata(title))
    return metadata


def _normalize_segmented_chunk(page_text: str, chunk_payload: dict) -> dict:
    chunk_text = chunk_payload["text"].strip()
    article_main = chunk_payload.get("article_main") or _primary_article_number(
        chunk_text
    )
    segmentation = chunk_payload.get("segmentation", "semantic_fallback")
    return {
        "text": chunk_text,
        "article_main": article_main,
        "article_references": _extract_article_references(chunk_text),
        "segmentation": segmentation,
        "page_is_context_heavy": _is_context_heavy_page(page_text),
    }


def _metadata_from_segment(
    document: dict,
    page_number: int,
    normalized_chunk: dict,
    page_used_ocr: bool,
    chunk_index: int,
    law_status: str,
) -> dict:
    article_main = normalized_chunk["article_main"]
    article_summary = _article_reference_summary(normalized_chunk["text"])
    metadata = build_official_metadata(
        document=document,
        page_number=page_number,
        chunk=normalized_chunk["text"],
        page_used_ocr=page_used_ocr,
        chunk_index=chunk_index,
        law_status=law_status,
        article_number=article_summary,
    )
    metadata["article_main"] = article_main
    metadata["article_references"] = normalized_chunk["article_references"]
    metadata["segmentation"] = normalized_chunk["segmentation"]
    metadata["page_is_context_heavy"] = normalized_chunk["page_is_context_heavy"]
    metadata["chunk_priority"] = _chunk_priority(
        article_main, normalized_chunk["segmentation"]
    )
    metadata["chunk_kind"] = _chunk_kind(normalized_chunk["segmentation"])
    if (
        metadata.get("page_is_context_heavy")
        and metadata.get("segmentation") != "article_block"
    ):
        metadata["source_priority"] = (
            float(metadata.get("source_priority", 0.0) or 0.0) - 0.15
        )
    return metadata


def _official_chunk_payload(
    document: dict,
    pdf_path: Path,
    page_number: int,
    normalized_chunk: dict,
    page_used_ocr: bool,
    chunk_index: int,
    law_status: str,
) -> dict:
    metadata = _metadata_from_segment(
        document=document,
        page_number=page_number,
        normalized_chunk=normalized_chunk,
        page_used_ocr=page_used_ocr,
        chunk_index=chunk_index,
        law_status=law_status,
    )
    chunk_text = normalized_chunk["text"]
    chunk_id = hashlib.sha1(
        f"{pdf_path.name}:{page_number}:{chunk_index}:{chunk_text[:120]}".encode(
            "utf-8"
        )
    ).hexdigest()
    return {"id": chunk_id, "text": chunk_text, "metadata": metadata}


def _page_chunks(page_text: str) -> list[dict]:
    return [
        _normalize_segmented_chunk(page_text, item)
        for item in legal_semantic_chunks(page_text)
    ]


def _preferred_ocr_page_range(pdf_path: Path, document: dict) -> tuple[int, int] | None:
    try:
        import fitz

        doc = fitz.open(pdf_path)
        if len(doc) < 3:
            return None
        first_lengths = [
            len((doc[i].get_text("text") or "").strip())
            for i in range(min(3, len(doc)))
        ]
        if all(length < 40 for length in first_lengths):
            return (2, len(doc))
    except Exception:
        return None
    return None


def _infer_law_status(filename: str, title: str | None = None) -> str:
    if title:
        spec = _document_spec(title)
        if spec and "law_status" in spec:
            return spec["law_status"]
    return "Não verificado"


def _download_enabled(document: dict) -> bool:
    return _source_url_for_document(document).lower().endswith(".pdf")


class OfficialLegislationDownloader:
    def __init__(self) -> None:
        self.settings = get_settings()

    def active_documents(self) -> list[dict]:
        return [
            dict(item) for item in OFFICIAL_DOCUMENT_CATALOG if _download_enabled(item)
        ]

    def skipped_documents(self) -> list[dict]:
        return [
            dict(item)
            for item in OFFICIAL_DOCUMENT_CATALOG
            if not _download_enabled(item)
        ]

    def download_all(self) -> list[dict]:
        downloaded: list[dict] = []
        skipped = self.skipped_documents()
        if skipped:
            for document in skipped:
                logger.warning(
                    "Catalogo oficial ignorado temporariamente por nao ser PDF directo: %s",
                    document["title"],
                )
        for document in self.active_documents():
            destination = _resolve_document_path(self.settings, document)
            if destination is None:
                continue
            downloaded.append(
                {
                    **document,
                    "path": destination,
                    "url": _source_url_for_document(document),
                }
            )
        logger.info(
            "Catalogo oficial preparado: total=%s activos=%s ignorados=%s",
            len(OFFICIAL_DOCUMENT_CATALOG),
            len(downloaded),
            len(skipped),
        )
        return downloaded

    def prepared_local_documents(self) -> list[dict]:
        prepared: list[dict] = []
        for document in OFFICIAL_DOCUMENT_CATALOG:
            destination = _resolve_document_path(self.settings, document)
            if destination is not None:
                prepared.append(
                    {
                        **document,
                        "path": destination,
                        "url": _source_url_for_document(document),
                    }
                )
        extra_local = self._extra_local_catalog_matches(prepared)
        prepared.extend(extra_local)
        return prepared

    def _extra_local_catalog_matches(self, existing_prepared: list[dict]) -> list[dict]:
        existing_titles = {item["title"] for item in existing_prepared}
        extras: list[dict] = []
        search_roots = [Path("C:/Projectos/TCC/legislacao"), self.settings.raw_pdfs_dir]
        for document in OFFICIAL_DOCUMENT_CATALOG:
            if document["title"] in existing_titles:
                continue
            target_tokens = {
                token
                for token in re.split(r"[^a-z0-9]+", document["title"].lower())
                if len(token) >= 4
            }
            found_path: Path | None = None
            for root in search_roots:
                if not root.exists():
                    continue
                for candidate in root.glob("*.pdf"):
                    haystack = candidate.name.lower()
                    overlap = sum(1 for token in target_tokens if token in haystack)
                    if overlap >= 2:
                        found_path = candidate
                        break
                if found_path:
                    break
            if found_path and found_path.exists() and found_path.stat().st_size > 0:
                copied = self.settings.raw_pdfs_dir / document["filename"]
                if found_path.resolve() != copied.resolve():
                    copied.write_bytes(found_path.read_bytes())
                extras.append({**document, "path": copied})
        return extras

    def missing_documents(self) -> list[dict]:
        present = {document["title"] for document in self.prepared_local_documents()}
        return [
            dict(document)
            for document in OFFICIAL_DOCUMENT_CATALOG
            if document["title"] not in present
        ]

    def prepare_documents(self) -> tuple[list[dict], list[dict]]:
        logger.info(
            "Preparando documentos oficiais (verificando arquivos locais e integridade)..."
        )
        downloaded = self.download_all()
        missing = self.missing_documents()
        if missing:
            logger.warning(
                "Catalogo oficial com lacunas locais: %s",
                ", ".join(document["title"] for document in missing),
            )
        return downloaded, missing

    def local_documents_for_reingest(self) -> list[dict]:
        prepared = self.prepared_local_documents()
        logger.info("Documentos locais disponiveis para reingestao: %s", len(prepared))
        return prepared

    def local_documents_for_slugs(self, diploma_slugs: set[str]) -> list[dict]:
        prepared = [
            item
            for item in self.prepared_local_documents()
            if item.get("diploma_slug") in diploma_slugs
        ]
        logger.info(
            "Documentos locais disponiveis para reingestao seletiva: %s", len(prepared)
        )
        return prepared

    def active_documents(self) -> list[dict]:
        return [
            dict(item) for item in OFFICIAL_DOCUMENT_CATALOG if _download_enabled(item)
        ]

    def skipped_documents(self) -> list[dict]:
        return [
            dict(item)
            for item in OFFICIAL_DOCUMENT_CATALOG
            if not _download_enabled(item)
        ]


class LegislationIngestionService:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def ingest_official_documents(self) -> IngestSummary:
        downloader = OfficialLegislationDownloader()
        documents, _missing = downloader.prepare_documents()
        processed_files: list[DocumentIngestResponse] = []
        total_chunks = 0

        logger.info("Iniciando ingestao de documentos oficiais. Resetando colecao...")
        # Comentado para permitir ingestão incremental/retomada
        # await legislation_vector_store.reset_collection()

        existing_slugs = legislation_vector_store.available_diploma_slugs()
        logger.info(
            "Retomando ingestao. Ja existem %s documentos. Processando %s documentos...",
            len(existing_slugs),
            len(documents),
        )

        for i, document in enumerate(documents, start=1):
            slug = document.get("diploma_slug")
            if slug and slug in existing_slugs:
                logger.info(
                    "[%s/%s] Ignorando (ja existe): %s",
                    i,
                    len(documents),
                    document.get("title"),
                )
                continue

            logger.info(
                "[%s/%s] Processando: %s", i, len(documents), document.get("title")
            )
            result = await self._ingest_document(document)
            processed_files.append(result)
            total_chunks += result.chunks_created
            logger.info(
                "Concluido: %s (%s chunks)",
                document.get("title"),
                result.chunks_created,
            )
            ocr_service.reset_engine()
            gc.collect()

        return IngestSummary(processed_files=processed_files, total_chunks=total_chunks)

    async def reingest_prepared_local_documents(self) -> IngestSummary:
        downloader = OfficialLegislationDownloader()
        documents = downloader.local_documents_for_reingest()
        processed_files: list[DocumentIngestResponse] = []
        total_chunks = 0

        await legislation_vector_store.reset_collection()

        for document in documents:
            result = await self._ingest_document(document)
            processed_files.append(result)
            total_chunks += result.chunks_created

        return IngestSummary(processed_files=processed_files, total_chunks=total_chunks)

    async def reingest_diploma_slugs(
        self, diploma_slugs: list[str], reset_collection: bool = False
    ) -> IngestSummary:
        downloader = OfficialLegislationDownloader()
        slug_set = {slug for slug in diploma_slugs if slug}
        processed_files: list[DocumentIngestResponse] = []
        total_chunks = 0

        if reset_collection:
            await legislation_vector_store.reset_collection()
        else:
            for slug in slug_set:
                removed = legislation_vector_store.delete_by_metadata(diploma_slug=slug)
                logger.info("Chunks removidos para %s: %s", slug, removed)

        documents = downloader.local_documents_for_slugs(slug_set)

        for document in documents:
            result = await self._ingest_document(document)
            processed_files.append(result)
            total_chunks += result.chunks_created

        return IngestSummary(processed_files=processed_files, total_chunks=total_chunks)

    async def _ingest_document(self, document: dict) -> DocumentIngestResponse:
        pdf_path: Path = document["path"]
        page_range = _preferred_ocr_page_range(pdf_path, document)
        if page_range:
            pages, used_ocr = extract_pages_from_pdf_range(
                pdf_path, first_page=page_range[0], last_page=page_range[1]
            )
        else:
            pages, used_ocr = extract_pages_from_pdf(pdf_path)
        ocr_pages = 0
        chunks_payload: list[dict] = []
        law_status = _infer_law_status(pdf_path.name, document["title"])

        for page_info in pages:
            page_number = page_info["page"]
            page_text = page_info["text"].strip()
            page_used_ocr = bool(page_info.get("used_ocr", False))
            if not page_text:
                continue
            if page_used_ocr:
                ocr_pages += 1
            for chunk_index, normalized_chunk in enumerate(
                _page_chunks(page_text), start=1
            ):
                chunks_payload.append(
                    _official_chunk_payload(
                        document=document,
                        pdf_path=pdf_path,
                        page_number=page_number,
                        normalized_chunk=normalized_chunk,
                        page_used_ocr=page_used_ocr,
                        chunk_index=chunk_index,
                        law_status=law_status,
                    )
                )

        chunk_count = await legislation_vector_store.upsert_documents(chunks_payload)
        logger.info(
            "Ficheiro %s processado, %s chunks criados, %s paginas via OCR",
            pdf_path.name,
            chunk_count,
            ocr_pages,
        )
        return DocumentIngestResponse(
            filename=pdf_path.name,
            chunks_created=chunk_count,
            used_ocr=used_ocr,
            ocr_pages=ocr_pages,
        )


legislation_ingestion_service = LegislationIngestionService()
