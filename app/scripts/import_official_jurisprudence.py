from __future__ import annotations

import argparse
import hashlib
import re
from html import unescape
from urllib.parse import urljoin, urlparse

import requests

from app.db.postgres import postgres_manager
from app.services.rag.embeddings import embedding_service

TS_INDEX_URL = "https://tribunalsupremo.ao/Categoria/jurisprudencia/"
TC_ARCHIVE_URL = "https://www.tribunalconstitucional.ao/pt/jurisprudencia/arquivo/"


def _clean(text: str) -> str:
    text = unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").casefold())
    return slug.strip("-")[:120] or "item"


def _normalize_supremo_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc in {"localhost", "www.localhost"}:
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return urljoin("https://tribunalsupremo.ao", path)
    return url


def _infer_branch_and_route(text: str) -> tuple[str, str]:
    haystack = (text or "").casefold()
    if any(token in haystack for token in ("laboral", "trabalho", "inss", "trabalhador")):
        return "laboral", "laboral"
    if any(token in haystack for token in ("tribut", "fiscal", "ipu", "iva")):
        return "tributario", "tributario"
    if any(token in haystack for token in ("acto administrativo", "ato administrativo", "contencioso")):
        return "administrativo", "contencioso_admin"
    if any(token in haystack for token in ("divórcio", "divorcio", "família", "familia")):
        return "familia", "familia"
    if any(token in haystack for token in ("constitucional", "inconstitucionalidade", "habeas corpus")):
        return "constitucional", "constitucional"
    if any(token in haystack for token in ("crime", "homicídio", "homicidio", "roubo", "violação", "violacao", "abuso sexual")):
        return "penal", "penal_substantivo"
    if any(token in haystack for token in ("sociedade", "acionista", "quotas", "providencia cautelar")):
        return "comercial", "sociedades"
    if any(token in haystack for token in ("herança", "heranca", "sucess")):
        return "familia", "sucessoes"
    return "indeterminado", "geral"


def _ts_cases_from_index(html: str) -> list[dict]:
    cases: list[dict] = []
    pattern = re.compile(
        r'<h2[^>]*>\s*<a[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>\s*</h2>',
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(html):
        title = _clean(match.group("title"))
        url = _normalize_supremo_url(urljoin(TS_INDEX_URL, match.group("url")))
        if not title:
            continue
        cases.append({"title": title, "url": url})
    return cases


def _ts_detail(detail_html: str) -> tuple[str | None, str | None, str | None]:
    title_match = re.search(r"<h1[^>]*>(.*?)</h1>", detail_html, re.IGNORECASE | re.DOTALL)
    title = _clean(title_match.group(1)) if title_match else None
    summary_match = re.search(
        r"Resumo do Acórdão:</strong>\s*(.*?)\s*</p>",
        detail_html,
        re.IGNORECASE | re.DOTALL,
    )
    summary = _clean(summary_match.group(1)) if summary_match else None
    pdf_match = re.search(r'href="([^"]+\.pdf[^"]*)".*?Descarregar', detail_html, re.IGNORECASE | re.DOTALL)
    pdf_url = (
        _normalize_supremo_url(urljoin(TS_INDEX_URL, pdf_match.group(1)))
        if pdf_match
        else None
    )
    return title, summary, pdf_url


def fetch_tribunal_supremo(limit: int = 20) -> list[dict]:
    response = requests.get(TS_INDEX_URL, timeout=30)
    response.raise_for_status()
    cases = _ts_cases_from_index(response.text)[:limit]
    enriched: list[dict] = []
    for item in cases:
        try:
            detail = requests.get(item["url"], timeout=30)
            detail.raise_for_status()
            title, summary, pdf_url = _ts_detail(detail.text)
        except Exception:
            title, summary, pdf_url = item["title"], None, None
        text = f"{title or item['title']} {summary or ''}".strip()
        branch, route = _infer_branch_and_route(text)
        case_id = "ts:" + hashlib.sha1(item["url"].encode("utf-8")).hexdigest()[:16]
        enriched.append(
            {
                "id": case_id,
                "court": "Tribunal Supremo",
                "chamber": None,
                "case_number": None,
                "title": title or item["title"],
                "publication_date": None,
                "url": item["url"],
                "pdf_url": pdf_url,
                "legal_branch": branch,
                "topic_route": route,
                "summary": summary,
                "metadata": {
                    "document_kind": "jurisprudence",
                    "source_kind": "jurisprudence",
                    "court": "Tribunal Supremo",
                },
            }
        )
    return enriched


def fetch_tribunal_constitucional(limit: int = 15) -> list[dict]:
    response = requests.get(TC_ARCHIVE_URL, timeout=30)
    response.raise_for_status()
    pattern = re.compile(
        r'<a[^>]+href="(?P<url>[^"]+)"[^>]*>\s*(?P<title>Ac[oó]rd[aã]o[^<]+)\s*</a>',
        re.IGNORECASE,
    )
    items: list[dict] = []
    for match in pattern.finditer(response.text):
        title = _clean(match.group("title"))
        url = urljoin(TC_ARCHIVE_URL, match.group("url"))
        if not title:
            continue
        case_id = "tc:" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        items.append(
            {
                "id": case_id,
                "court": "Tribunal Constitucional",
                "chamber": None,
                "case_number": None,
                "title": title,
                "publication_date": None,
                "url": url,
                "pdf_url": url if url.lower().endswith(".pdf") else None,
                "legal_branch": "constitucional",
                "topic_route": "constitucional",
                "summary": f"{title}. Jurisprudência oficial do Tribunal Constitucional de Angola.",
                "metadata": {
                    "document_kind": "jurisprudence",
                    "source_kind": "jurisprudence",
                    "court": "Tribunal Constitucional",
                },
            }
        )
        if len(items) >= limit:
            break
    return items


async def _to_segments(cases: list[dict]) -> list[dict]:
    segments: list[dict] = []
    for item in cases:
        summary = item.get("summary") or item["title"]
        text = f"{item['title']}. {summary}".strip()
        embedding = await embedding_service.embed_query(text)
        slug = f"jurisprudencia-{item['court'].casefold().replace(' ', '-')}-{_slugify(item['title'])}"
        metadata = {
            "source": item["court"],
            "title": item["title"],
            "link_original": item["url"],
            "page": None,
            "article_number": None,
            "article_main": None,
            "article_references": [],
            "law_status": "Jurisprudência oficial",
            "source_scope": "official",
            "document_id": item["id"],
            "diploma_slug": slug,
            "legal_branch": item.get("legal_branch") or "indeterminado",
            "topic_route": item.get("topic_route") or "geral",
            "document_kind": "jurisprudence",
            "source_kind": "jurisprudence",
            "court": item["court"],
        }
        segments.append(
            {
                "id": f"{item['id']}:summary",
                "text": text,
                "embedding": embedding,
                "metadata": metadata,
            }
        )
    return segments


async def main(limit_ts: int, limit_tc: int) -> None:
    postgres_manager.initialize()
    cases = fetch_tribunal_supremo(limit_ts) + fetch_tribunal_constitucional(limit_tc)
    postgres_manager.upsert_jurisprudence_cases(cases)
    postgres_manager.upsert_legal_segments(await _to_segments(cases))
    print(f"Importados {len(cases)} registos de jurisprudência oficial.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-ts", type=int, default=20)
    parser.add_argument("--limit-tc", type=int, default=15)
    args = parser.parse_args()

    import asyncio

    asyncio.run(main(args.limit_ts, args.limit_tc))
