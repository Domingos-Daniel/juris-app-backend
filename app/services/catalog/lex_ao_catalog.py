from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from app.core.logger import get_logger


logger = get_logger(__name__)

SITEMAP_URL = "https://lex.ao/sitemap.xml"
DOC_PREFIX = "https://lex.ao/docs/"
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
META_NAME_RE = re.compile(r'<meta[^>]+name=(?:"|\')(?P<name>[^"\']+)(?:"|\')[^>]+content=(?:"|\')(?P<content>[^"\']*)(?:"|\')', re.IGNORECASE)
META_PROPERTY_RE = re.compile(r'<meta[^>]+property=(?:"|\')(?P<name>[^"\']+)(?:"|\')[^>]+content=(?:"|\')(?P<content>[^"\']*)(?:"|\')', re.IGNORECASE)
DOWNLOAD_RE = re.compile(r'href=(?P<quote>["\']?)(?P<href>https://files\.lex\.ao/[^"\' >]+/download/[^"\' >]+\.pdf)(?P=quote)', re.IGNORECASE)
LAW_SIGNATURE_RE = re.compile(r"\b(?P<kind>lei|decreto|acord[aã]o|resolu[cç][aã]o)\s+n\.\s*o\s*(?P<number>\d+)\/(?P<year>\d{2,4})\b", re.IGNORECASE)

BRANCH_HINTS = {
    "familia": "familia",
    "bilhete": "administrativo",
    "identidade": "administrativo",
    "sociedade": "comercial",
    "sociedades": "comercial",
    "contencioso": "administrativo",
    "trabalho": "laboral",
    "penal": "penal",
    "tribunal": "administrativo",
}

TOPIC_ROUTE_HINTS = {
    "familia": "familia",
    "herança": "familia",
    "heranca": "familia",
    "bilhete": "identificacao_civil",
    "identidade": "identificacao_civil",
    "sociedade": "sociedades",
    "sociedades": "sociedades",
    "contencioso": "contencioso_admin",
    "trabalho": "laboral",
    "penal": "penal_substantivo",
}

KNOWN_DIPLOMA_MATCHES = {
    "código da família": "codigo-familia-lei-1-88",
    "codigo da familia": "codigo-familia-lei-1-88",
    "lei do bilhete de identidade": "lei-bilhete-identidade-4-16",
    "lei das sociedades comerciais": "lei-sociedades-comerciais-1-04",
    "lei do contencioso administrativo": "codigo-processo-contencioso-administrativo-33-22",
    "código de processo do contencioso administrativo": "codigo-processo-contencioso-administrativo-33-22",
    "codigo de processo do contencioso administrativo": "codigo-processo-contencioso-administrativo-33-22",
    "código penal": "codigo-penal-lei-38-20",
    "codigo penal": "codigo-penal-lei-38-20",
    "código do processo penal": "codigo-processo-penal-lei-39-20",
    "codigo do processo penal": "codigo-processo-penal-lei-39-20",
    "lei geral do trabalho": "lei-geral-do-trabalho-lei-12-23",
    "código civil": "codigo-civil",
    "codigo civil": "codigo-civil",
}

KNOWN_SIGNATURE_MATCHES = {
    ("assembleia-do-povo", "lei", "1", "88"): "codigo-familia-lei-1-88",
    ("assembleia-nacional", "lei", "4", "16"): "lei-bilhete-identidade-4-16",
    ("assembleia-nacional", "lei", "1", "04"): "lei-sociedades-comerciais-1-04",
    ("assembleia-nacional", "lei", "33", "22"): "codigo-processo-contencioso-administrativo-33-22",
}

KNOWN_ROUTE_SLUG_MATCHES = {
    "lei-n-o-1-88-de-20-de-fevereiro": "codigo-familia-lei-1-88",
    "lei-n-o-4-16-de-17-de-maio": "lei-bilhete-identidade-4-16",
    "lei-n-o-1-04-de-13-de-fevereiro": "lei-sociedades-comerciais-1-04",
    "lei-n-o-33-22-de-01-de-setembro": "codigo-processo-contencioso-administrativo-33-22",
}

PRIORITY_PAGE_URLS = {
    "codigo-familia-lei-1-88": "https://lex.ao/docs/assembleia-do-povo/1988/lei-n-o-1-88-de-20-de-fevereiro/",
    "lei-bilhete-identidade-4-16": "https://lex.ao/docs/assembleia-nacional/2016/lei-n-o-4-16-de-17-de-maio/",
    "lei-sociedades-comerciais-1-04": "https://lex.ao/docs/assembleia-nacional/2004/lei-n-o-1-04-de-13-de-fevereiro/",
    "codigo-processo-contencioso-administrativo-33-22": "https://lex.ao/docs/assembleia-nacional/2022/lei-n-o-33-22-de-01-de-setembro/",
}


@dataclass(slots=True)
class LexAoDocument:
    page_url: str
    entity_slug: str
    entity_name: str
    year: str
    document_slug: str
    title: str
    description: str
    keywords: list[str]
    download_pdf_url: str | None
    matched_internal_slug: str | None
    legal_branch_guess: str
    topic_route_guess: str

    def to_dict(self) -> dict:
        return {
            "page_url": self.page_url,
            "entity_slug": self.entity_slug,
            "entity_name": self.entity_name,
            "year": self.year,
            "document_slug": self.document_slug,
            "title": self.title,
            "description": self.description,
            "keywords": self.keywords,
            "download_pdf_url": self.download_pdf_url,
            "matched_internal_slug": self.matched_internal_slug,
            "legal_branch_guess": self.legal_branch_guess,
            "topic_route_guess": self.topic_route_guess,
        }


class LexAoCatalogService:
    def __init__(self) -> None:
        self.base_dir = Path("C:/Projectos/TCC/backend/data/catalogs")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.sitemap_cache = self.base_dir / "lex_ao_sitemap.json"
        self.documents_cache = self.base_dir / "lex_ao_documents.json"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TCC-LexCatalog/1.0"})

    def fetch_sitemap_urls(self) -> list[str]:
        xml_text = self.session.get(SITEMAP_URL, timeout=60).text
        root = ET.fromstring(xml_text)
        urls = [node.text.strip() for node in root.findall("sm:url/sm:loc", NS) if node.text and node.text.startswith(DOC_PREFIX)]
        self.sitemap_cache.write_text(json.dumps(urls, ensure_ascii=False, indent=2), encoding="utf-8")
        return urls

    def parse_document_page(self, page_url: str) -> LexAoDocument | None:
        parsed = urlparse(page_url)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 4 or parts[0] != "docs":
            return None
        entity_slug, year, document_slug = parts[1], parts[2], parts[3]
        html = self.session.get(page_url, timeout=60).text
        title = self._extract_title(html)
        description = self._extract_meta(html, "description")
        keywords_text = self._extract_meta(html, "keywords")
        keywords = [item.strip() for item in keywords_text.split(",") if item.strip()]
        download_url = self._extract_download_url(html)
        entity_name = self._humanize_slug(entity_slug)
        matched_slug = self._match_internal_slug(
            title=title,
            description=description,
            entity_slug=entity_slug,
            year=year,
            document_slug=document_slug,
        )
        branch_guess = self._guess_branch(title, description, entity_name)
        topic_route_guess = self._guess_topic_route(title, description)
        return LexAoDocument(
            page_url=page_url,
            entity_slug=entity_slug,
            entity_name=entity_name,
            year=year,
            document_slug=document_slug,
            title=title,
            description=description,
            keywords=keywords,
            download_pdf_url=download_url,
            matched_internal_slug=matched_slug,
            legal_branch_guess=branch_guess,
            topic_route_guess=topic_route_guess,
        )

    def build_catalog(self, limit: int | None = None) -> dict:
        urls = self.fetch_sitemap_urls()
        documents: list[dict] = []
        processed = 0
        for url in urls:
            if limit is not None and processed >= limit:
                break
            try:
                parsed = self.parse_document_page(url)
                if parsed:
                    documents.append(parsed.to_dict())
                    processed += 1
            except Exception as exc:
                logger.warning("Falha a analisar %s: %s", url, exc)
        payload = {
            "summary": self._summarize(documents),
            "documents": documents,
        }
        self.documents_cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def priority_targets(self, payload: dict | None = None) -> list[dict]:
        if payload is None:
            payload = json.loads(self.documents_cache.read_text(encoding="utf-8")) if self.documents_cache.exists() else {"documents": []}
        wanted = {
            "codigo-familia-lei-1-88",
            "lei-bilhete-identidade-4-16",
            "lei-sociedades-comerciais-1-04",
            "codigo-processo-contencioso-administrativo-33-22",
        }
        found: dict[str, dict] = {
            item["matched_internal_slug"]: item
            for item in payload.get("documents", [])
            if item.get("matched_internal_slug") in wanted
        }
        missing = wanted.difference(found)
        for slug in sorted(missing):
            page_url = PRIORITY_PAGE_URLS.get(slug)
            if not page_url:
                continue
            try:
                parsed = self.parse_document_page(page_url)
            except Exception as exc:
                logger.warning("Falha ao resolver diploma prioritario %s: %s", slug, exc)
                continue
            if parsed and parsed.matched_internal_slug in wanted:
                found[parsed.matched_internal_slug] = parsed.to_dict()
        return [found[slug] for slug in sorted(found)]

    def _extract_title(self, html: str) -> str:
        match = TITLE_RE.search(html)
        if not match:
            return ""
        title = re.sub(r"\s*\|.*$", "", match.group(1)).strip()
        return re.sub(r"\s+", " ", title)

    def _extract_meta(self, html: str, name: str) -> str:
        for match in META_NAME_RE.finditer(html):
            if match.group("name").strip().lower() == name.lower():
                return match.group("content").strip()
        for match in META_PROPERTY_RE.finditer(html):
            if match.group("name").strip().lower() == name.lower():
                return match.group("content").strip()
        return ""

    def _extract_download_url(self, html: str) -> str | None:
        match = DOWNLOAD_RE.search(html)
        if not match:
            return None
        href = match.group("href")
        return urljoin("https://lex.ao", href)

    def _humanize_slug(self, slug: str) -> str:
        return re.sub(r"\s+", " ", slug.replace("-", " ")).strip().title()

    def _match_internal_slug(
        self,
        title: str,
        description: str,
        entity_slug: str,
        year: str,
        document_slug: str,
    ) -> str | None:
        haystack = f"{title} {description}".lower()
        for key, slug in KNOWN_DIPLOMA_MATCHES.items():
            if key in haystack:
                return slug

        route_match = KNOWN_ROUTE_SLUG_MATCHES.get(document_slug.lower())
        if route_match:
            return route_match

        signature = self._extract_law_signature(title)
        if signature:
            kind, number, year_short = signature
            exact = KNOWN_SIGNATURE_MATCHES.get((entity_slug.lower(), kind, number, year_short))
            if exact:
                return exact

            normalized_year = year[-2:] if len(year) >= 2 else year
            if normalized_year == year_short:
                exact = KNOWN_SIGNATURE_MATCHES.get((entity_slug.lower(), kind, number, normalized_year))
                if exact:
                    return exact
        return None

    def _extract_law_signature(self, title: str) -> tuple[str, str, str] | None:
        match = LAW_SIGNATURE_RE.search(title)
        if not match:
            return None
        kind = match.group("kind").strip().lower()
        number = match.group("number").lstrip("0") or "0"
        raw_year = match.group("year")
        year = raw_year[-2:]
        return kind, number, year

    def _guess_branch(self, title: str, description: str, entity_name: str) -> str:
        haystack = f"{title} {description} {entity_name}".lower()
        for key, branch in BRANCH_HINTS.items():
            if key in haystack:
                return branch
        return "indeterminado"

    def _guess_topic_route(self, title: str, description: str) -> str:
        haystack = f"{title} {description}".lower()
        for key, route in TOPIC_ROUTE_HINTS.items():
            if key in haystack:
                return route
        return "geral"

    def _summarize(self, documents: list[dict]) -> dict:
        total = len(documents)
        with_pdf = sum(1 for item in documents if item.get("download_pdf_url"))
        matched = sum(1 for item in documents if item.get("matched_internal_slug"))
        branches: dict[str, int] = {}
        for item in documents:
            branch = item.get("legal_branch_guess") or "indeterminado"
            branches[branch] = branches.get(branch, 0) + 1
        return {
            "total_documents": total,
            "with_download_pdf": with_pdf,
            "matched_internal_slug": matched,
            "branches": branches,
        }


lex_ao_catalog_service = LexAoCatalogService()
