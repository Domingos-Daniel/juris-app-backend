from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from cachetools import TTLCache

from app.core.config import get_settings
from app.core.logger import get_logger
from app.db.models import RetrievedChunk


logger = get_logger(__name__)

# Cache de embeddings por query string para evitar geração redundante.
# Cada chamada a retriever_service.retrieve() gera um embedding mesmo para
# variantes da mesma query. Com ~20 queries por pedido, poupa ~2s.
_embedding_cache: TTLCache = TTLCache(maxsize=64, ttl=60)

DEFAULT_USER_ID = "default-user"
DEFAULT_USER_NAME = "Utilizador Local"

# Portuguese stopwords for FTS filter building
_FTS_STOPWORDS = frozenset(
    {
        "de",
        "da",
        "do",
        "das",
        "dos",
        "e",
        "ou",
        "a",
        "o",
        "as",
        "os",
        "um",
        "uma",
        "no",
        "na",
        "nos",
        "nas",
        "com",
        "sem",
        "por",
        "para",
        "que",
        "quando",
        "como",
        "entre",
        "sobre",
        "agora",
        "mesmo",
        "caso",
        "se",
        "ao",
        "aos",
        "em",
        "pelo",
        "pela",
        "pelos",
        "pelas",
        "mais",
        "menos",
        "muito",
        "pouco",
        "todo",
        "toda",
    }
)
_FTS_WORD_RE = __import__("re").compile(r"\w+", __import__("re").UNICODE)


def _build_fts_or_query(query: str) -> str | None:
    """Build an OR-based tsquery from individual query terms.

    Uses ANY-term match (OR logic) so the GIN index can narrow the scan
    without being overly restrictive. Caps at 12 terms to avoid bloat.
    """
    matches = _FTS_WORD_RE.findall(query.lower())
    terms = [t for t in matches if t not in _FTS_STOPWORDS and len(t) >= 2][:12]
    if not terms:
        return None
    return " | ".join(terms)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _vector_literal(values: list[float] | None) -> str | None:
    if not values:
        return None
    return "{" + ",".join(f"{float(value):.12g}" for value in values) + "}"


class PostgresManager:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._initialized = False
        self._pool: ConnectionPool | None = None

    def _require_dsn(self) -> str:
        dsn = (self.settings.postgres_dsn or "").strip()
        if not dsn:
            raise RuntimeError(
                "POSTGRES_DSN não está configurado. Este backend já não suporta SQLite."
            )
        return dsn

    def _get_pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(
                self._require_dsn(),
                min_size=2,
                max_size=10,
                kwargs={"row_factory": dict_row, "connect_timeout": 5},
            )
        return self._pool

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            pool.putconn(conn)

    def initialize(self) -> None:
        if self._initialized:
            return
        schema = self.settings.postgres_schema
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
                cur.execute(f"SET search_path TO {schema}, public")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT,
                        is_seeded BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS auth_tokens (
                        token TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        username TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chats (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        title TEXT NOT NULL,
                        active_document_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id TEXT PRIMARY KEY,
                        chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        provider_used TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        sources_json JSONB NOT NULL DEFAULT '[]'::jsonb
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        filename TEXT NOT NULL,
                        display_name TEXT NOT NULL,
                        storage_path TEXT,
                        mime_type TEXT NOT NULL,
                        size_bytes BIGINT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        page_count INTEGER NOT NULL DEFAULT 0,
                        chunks_created INTEGER NOT NULL DEFAULT 0,
                        extraction_mode TEXT NOT NULL DEFAULT 'direct',
                        quality_status TEXT NOT NULL DEFAULT 'good',
                        summary TEXT,
                        preview_text TEXT,
                        category TEXT,
                        last_used_at TIMESTAMPTZ
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS queries (
                        id BIGSERIAL PRIMARY KEY,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS legal_documents (
                        id TEXT PRIMARY KEY,
                        entity_slug TEXT,
                        entity_name TEXT,
                        year TEXT,
                        document_slug TEXT,
                        title TEXT NOT NULL,
                        page_url TEXT,
                        download_pdf_url TEXT,
                        matched_internal_slug TEXT,
                        legal_branch_guess TEXT,
                        topic_route_guess TEXT,
                        status TEXT NOT NULL DEFAULT 'discovered',
                        source_invalid BOOLEAN NOT NULL DEFAULT FALSE,
                        local_pdf_path TEXT,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS legal_document_versions (
                        id TEXT PRIMARY KEY,
                        legal_document_id TEXT NOT NULL REFERENCES legal_documents(id) ON DELETE CASCADE,
                        source_hash TEXT,
                        version_label TEXT,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS legal_segments (
                        id TEXT PRIMARY KEY,
                        legal_document_id TEXT REFERENCES legal_documents(id) ON DELETE CASCADE,
                        source TEXT NOT NULL,
                        title TEXT NOT NULL,
                        link_original TEXT,
                        page INTEGER,
                        article_number TEXT,
                        article_main TEXT,
                        article_references TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                        law_status TEXT NOT NULL DEFAULT 'Nao verificado',
                        source_scope TEXT NOT NULL DEFAULT 'official',
                        document_id TEXT,
                        diploma_slug TEXT,
                        legal_branch TEXT,
                        topic_route TEXT,
                        text_content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        text_search TSVECTOR,
                        embedding double precision[],
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jurisprudence_cases (
                        id TEXT PRIMARY KEY,
                        court TEXT NOT NULL,
                        chamber TEXT,
                        case_number TEXT,
                        title TEXT NOT NULL,
                        decision_date DATE,
                        publication_date DATE,
                        url TEXT NOT NULL,
                        pdf_url TEXT,
                        legal_branch TEXT,
                        topic_route TEXT,
                        summary TEXT,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jurisprudence_holdings (
                        id TEXT PRIMARY KEY,
                        jurisprudence_case_id TEXT NOT NULL REFERENCES jurisprudence_cases(id) ON DELETE CASCADE,
                        holding_text TEXT NOT NULL,
                        legal_branch TEXT,
                        topic_route TEXT,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS legal_citations (
                        id TEXT PRIMARY KEY,
                        legal_segment_id TEXT NOT NULL REFERENCES legal_segments(id) ON DELETE CASCADE,
                        citation_text TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS legal_relations (
                        id TEXT PRIMARY KEY,
                        source_document_id TEXT NOT NULL REFERENCES legal_documents(id) ON DELETE CASCADE,
                        target_document_id TEXT REFERENCES legal_documents(id) ON DELETE CASCADE,
                        relation_type TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ingestion_jobs (
                        id TEXT PRIMARY KEY,
                        job_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ingestion_job_items (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES ingestion_jobs(id) ON DELETE CASCADE,
                        legal_document_id TEXT REFERENCES legal_documents(id) ON DELETE CASCADE,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_legal_state (
                        chat_id TEXT PRIMARY KEY REFERENCES chats(id) ON DELETE CASCADE,
                        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        topic_route TEXT,
                        legal_branch TEXT,
                        diploma_slug TEXT,
                        active_article TEXT,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chats_user_updated ON chats(user_id, updated_at DESC)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_created ON chat_messages(chat_id, created_at ASC)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_user_created ON documents(user_id, created_at DESC)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_legal_documents_slug ON legal_documents(matched_internal_slug, document_slug)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_legal_segments_lookup ON legal_segments(diploma_slug, legal_branch, page)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_legal_segments_doc ON legal_segments(document_id, source_scope)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_jurisprudence_cases_branch ON jurisprudence_cases(court, legal_branch, publication_date DESC)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_legal_segments_fts ON legal_segments USING GIN(text_search)"
                )
            self._seed_default_user(conn)
        self._initialized = True

    def _seed_default_user(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (id, name, email, is_seeded, created_at)
                VALUES (%s, %s, %s, TRUE, %s)
                ON CONFLICT (id) DO UPDATE
                SET name = EXCLUDED.name,
                    is_seeded = TRUE
                """,
                (DEFAULT_USER_ID, DEFAULT_USER_NAME, None, utc_now_iso()),
            )

    def issue_auth_token(self, user_id: str, username: str, token: str) -> None:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO auth_tokens (token, user_id, username, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (token) DO UPDATE
                SET username = EXCLUDED.username,
                    created_at = EXCLUDED.created_at
                """,
                (token, user_id, username, utc_now_iso()),
            )

    def has_auth_token(self, token: str) -> bool:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM auth_tokens WHERE token = %s", (token,))
            return cur.fetchone() is not None

    def get_default_user_id(self) -> str:
        self.initialize()
        return DEFAULT_USER_ID

    def get_default_user_name(self) -> str:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT name FROM users WHERE id = %s", (DEFAULT_USER_ID,))
            row = cur.fetchone()
            return row["name"] if row and row.get("name") else DEFAULT_USER_NAME

    def save_query(self, question: str, answer: str) -> int:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO queries (question, answer, timestamp) VALUES (%s, %s, %s) RETURNING id",
                (question, answer, utc_now_iso()),
            )
            row = cur.fetchone()
            return int(row["id"])

    def create_chat(
        self,
        title: str,
        active_document_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        self.initialize()
        chat_id = str(uuid.uuid4())
        now = utc_now_iso()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chats (id, user_id, title, active_document_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    chat_id,
                    user_id or DEFAULT_USER_ID,
                    title,
                    active_document_id,
                    now,
                    now,
                ),
            )
        return chat_id

    def append_chat_exchange(
        self,
        *,
        chat_id: str,
        question: str,
        answer: str,
        provider_used: str,
        sources: list[dict],
        active_document_id: str | None = None,
    ) -> None:
        self.initialize()
        now_user = utc_now_iso()
        now_assistant = utc_now_iso()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chats
                SET updated_at = %s,
                    active_document_id = COALESCE(%s, active_document_id)
                WHERE id = %s
                """,
                (now_assistant, active_document_id, chat_id),
            )
            cur.execute(
                """
                INSERT INTO chat_messages (id, chat_id, role, content, provider_used, created_at, sources_json)
                VALUES (%s, %s, 'user', %s, NULL, %s, '[]'::jsonb)
                """,
                (str(uuid.uuid4()), chat_id, question, now_user),
            )
            cur.execute(
                """
                INSERT INTO chat_messages (id, chat_id, role, content, provider_used, created_at, sources_json)
                VALUES (%s, %s, 'assistant', %s, %s, %s, %s::jsonb)
                """,
                (
                    str(uuid.uuid4()),
                    chat_id,
                    answer,
                    provider_used,
                    now_assistant,
                    json.dumps(sources, ensure_ascii=False),
                ),
            )

    def get_conversation_state(
        self, chat_id: str, user_id: str | int | None = None
    ) -> dict[str, Any] | None:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chat_id, user_id, topic_route, legal_branch, diploma_slug, active_article, metadata, updated_at
                FROM conversation_legal_state
                WHERE chat_id = %s AND user_id = %s
                """,
                (chat_id, user_id or DEFAULT_USER_ID),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "chat_id": row["chat_id"],
                "user_id": row["user_id"],
                "topic_route": row["topic_route"],
                "legal_branch": row["legal_branch"],
                "diploma_slug": row["diploma_slug"],
                "active_article": row["active_article"],
                "metadata": row["metadata"] or {},
                "updated_at": row["updated_at"].isoformat(),
            }

    def upsert_conversation_state(
        self,
        *,
        chat_id: str,
        user_id: str | int | None = None,
        topic_route: str | None = None,
        legal_branch: str | None = None,
        diploma_slug: str | None = None,
        active_article: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversation_legal_state (
                    chat_id, user_id, topic_route, legal_branch, diploma_slug, active_article, metadata, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (chat_id) DO UPDATE
                SET user_id = EXCLUDED.user_id,
                    topic_route = EXCLUDED.topic_route,
                    legal_branch = EXCLUDED.legal_branch,
                    diploma_slug = EXCLUDED.diploma_slug,
                    active_article = EXCLUDED.active_article,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    chat_id,
                    user_id or DEFAULT_USER_ID,
                    topic_route,
                    legal_branch,
                    diploma_slug,
                    active_article,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    utc_now_iso(),
                ),
            )

    def list_chats(self, user_id: str | None = None) -> list[dict]:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, created_at, updated_at, active_document_id
                FROM chats
                WHERE user_id = %s
                ORDER BY updated_at DESC
                """,
                (user_id or DEFAULT_USER_ID,),
            )
            chats = cur.fetchall()
            items: list[dict] = []
            for chat in chats:
                cur.execute(
                    """
                    SELECT id, role, content, provider_used, created_at, sources_json
                    FROM chat_messages
                    WHERE chat_id = %s
                    ORDER BY created_at ASC, id ASC
                    """,
                    (chat["id"],),
                )
                messages = cur.fetchall()
                items.append(
                    {
                        "id": chat["id"],
                        "title": chat["title"],
                        "created_at": chat["created_at"].isoformat(),
                        "updated_at": chat["updated_at"].isoformat(),
                        "active_document_id": chat["active_document_id"],
                        "messages": [
                            {
                                "id": message["id"],
                                "role": message["role"],
                                "content": message["content"],
                                "provider_used": message["provider_used"],
                                "created_at": message["created_at"].isoformat(),
                                "sources": message["sources_json"] or [],
                            }
                            for message in messages
                        ],
                    }
                )
            return items

    def delete_chat(self, chat_id: str, user_id: str | None = None) -> bool:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            params = [chat_id]
            where = "id = %s"
            if user_id:
                where += " AND user_id = %s"
                params.append(user_id)
            cur.execute(f"DELETE FROM chats WHERE {where}", params)
            return cur.rowcount > 0

    def delete_all_chats(self, user_id: str | None = None) -> int:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            if user_id:
                cur.execute("DELETE FROM chats WHERE user_id = %s", (user_id,))
            else:
                cur.execute("DELETE FROM chats")
            return cur.rowcount

    def save_document(
        self,
        *,
        filename: str,
        mime_type: str,
        size_bytes: int,
        status: str,
        page_count: int,
        chunks_created: int,
        extraction_mode: str,
        display_name: str | None = None,
        storage_path: str | None = None,
        quality_status: str = "good",
        summary: str | None = None,
        preview_text: str | None = None,
        category: str | None = None,
        user_id: str | None = None,
    ) -> str:
        self.initialize()
        document_id = str(uuid.uuid4())
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (
                    id, user_id, filename, display_name, storage_path, mime_type,
                    size_bytes, status, created_at, page_count, chunks_created,
                    extraction_mode, quality_status, summary, preview_text, category
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    user_id or DEFAULT_USER_ID,
                    filename,
                    display_name or filename,
                    storage_path,
                    mime_type,
                    size_bytes,
                    status,
                    utc_now_iso(),
                    page_count,
                    chunks_created,
                    extraction_mode,
                    quality_status,
                    summary,
                    preview_text,
                    category,
                ),
            )
        return document_id

    def rename_document(
        self, document_id: str, display_name: str, user_id: str | int | None = None
    ) -> None:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET display_name = %s WHERE id = %s AND user_id = %s",
                (display_name, document_id, user_id or DEFAULT_USER_ID),
            )

    def mark_document_used(
        self, document_id: str, user_id: str | int | None = None
    ) -> None:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET last_used_at = %s WHERE id = %s AND user_id = %s",
                (utc_now_iso(), document_id, user_id or DEFAULT_USER_ID),
            )

    def delete_document(
        self, document_id: str, user_id: str | int | None = None
    ) -> None:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE chats SET active_document_id = NULL WHERE active_document_id = %s AND user_id = %s",
                (document_id, user_id or DEFAULT_USER_ID),
            )
            cur.execute(
                "DELETE FROM documents WHERE id = %s AND user_id = %s",
                (document_id, user_id or DEFAULT_USER_ID),
            )

    def get_document(
        self, document_id: str, user_id: str | int | None = None
    ) -> dict | None:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, display_name, storage_path, mime_type, size_bytes, status,
                       created_at, page_count, chunks_created, extraction_mode, quality_status,
                       summary, preview_text, category, last_used_at
                FROM documents
                WHERE id = %s AND user_id = %s
                """,
                (document_id, user_id or DEFAULT_USER_ID),
            )
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(
                """
                SELECT COUNT(*) AS usage_count, MAX(updated_at) AS last_chat_use
                FROM chats
                WHERE active_document_id = %s AND user_id = %s
                """,
                (document_id, user_id or DEFAULT_USER_ID),
            )
            usage = cur.fetchone() or {"usage_count": 0, "last_chat_use": None}
            return {
                "id": row["id"],
                "filename": row["filename"],
                "display_name": row["display_name"],
                "storage_path": row["storage_path"],
                "mime_type": row["mime_type"],
                "size_bytes": int(row["size_bytes"]),
                "status": row["status"],
                "created_at": row["created_at"].isoformat(),
                "page_count": int(row["page_count"] or 0),
                "chunks_created": int(row["chunks_created"] or 0),
                "extraction_mode": row["extraction_mode"],
                "quality_status": row["quality_status"],
                "summary": row["summary"],
                "preview_text": row["preview_text"],
                "category": row["category"],
                "usage_count": int(usage["usage_count"] or 0),
                "last_used_at": (
                    row["last_used_at"].isoformat()
                    if row["last_used_at"]
                    else (
                        usage["last_chat_use"].isoformat()
                        if usage["last_chat_use"]
                        else None
                    )
                ),
            }

    def set_chat_active_document(
        self, chat_id: str, document_id: str | None, user_id: str | int | None = None
    ) -> None:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE chats SET active_document_id = %s, updated_at = %s WHERE id = %s AND user_id = %s",
                (document_id, utc_now_iso(), chat_id, user_id or DEFAULT_USER_ID),
            )

    def list_documents(self, user_id: str | None = None) -> list[dict]:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, display_name, storage_path, mime_type, size_bytes, status,
                       created_at, page_count, chunks_created, extraction_mode, quality_status,
                       summary, preview_text, category, last_used_at
                FROM documents
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id or DEFAULT_USER_ID,),
            )
            rows = cur.fetchall()
            items: list[dict] = []
            for row in rows:
                cur.execute(
                    "SELECT COUNT(*) AS usage_count, MAX(updated_at) AS last_chat_use FROM chats WHERE active_document_id = %s AND user_id = %s",
                    (row["id"], user_id or DEFAULT_USER_ID),
                )
                usage = cur.fetchone() or {"usage_count": 0, "last_chat_use": None}
                items.append(
                    {
                        "id": row["id"],
                        "filename": row["filename"],
                        "display_name": row["display_name"],
                        "storage_path": row["storage_path"],
                        "mime_type": row["mime_type"],
                        "size_bytes": int(row["size_bytes"]),
                        "status": row["status"],
                        "created_at": row["created_at"].isoformat(),
                        "page_count": int(row["page_count"] or 0),
                        "chunks_created": int(row["chunks_created"] or 0),
                        "extraction_mode": row["extraction_mode"],
                        "quality_status": row["quality_status"],
                        "summary": row["summary"],
                        "preview_text": row["preview_text"],
                        "category": row["category"],
                        "usage_count": int(usage["usage_count"] or 0),
                        "last_used_at": (
                            row["last_used_at"].isoformat()
                            if row["last_used_at"]
                            else (
                                usage["last_chat_use"].isoformat()
                                if usage["last_chat_use"]
                                else None
                            )
                        ),
                    }
                )
            return items

    def import_legal_documents(
        self,
        documents: list[dict[str, Any]],
        local_path_resolver: callable | None = None,
    ) -> int:
        self.initialize()
        imported = 0
        with self.connection() as conn, conn.cursor() as cur:
            for doc in documents:
                local_path = local_path_resolver(doc) if local_path_resolver else None
                source_invalid = bool(doc.get("source_invalid", False))
                if local_path and not Path(local_path).exists():
                    local_path = None
                now = utc_now_iso()
                doc_id = f"lexao:{doc.get('entity_slug', 'unknown')}:{doc.get('year', 'unknown')}:{doc.get('document_slug', 'unknown')}"
                cur.execute(
                    """
                    INSERT INTO legal_documents (
                        id, entity_slug, entity_name, year, document_slug, title, page_url,
                        download_pdf_url, matched_internal_slug, legal_branch_guess,
                        topic_route_guess, status, source_invalid, local_pdf_path,
                        metadata, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET title = EXCLUDED.title,
                        download_pdf_url = EXCLUDED.download_pdf_url,
                        matched_internal_slug = EXCLUDED.matched_internal_slug,
                        legal_branch_guess = EXCLUDED.legal_branch_guess,
                        topic_route_guess = EXCLUDED.topic_route_guess,
                        status = EXCLUDED.status,
                        source_invalid = EXCLUDED.source_invalid,
                        local_pdf_path = EXCLUDED.local_pdf_path,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        doc_id,
                        doc.get("entity_slug"),
                        doc.get("entity_name"),
                        doc.get("year"),
                        doc.get("document_slug"),
                        doc.get("title") or "Documento",
                        doc.get("page_url"),
                        doc.get("download_pdf_url"),
                        doc.get("matched_internal_slug"),
                        doc.get("legal_branch_guess"),
                        doc.get("topic_route_guess"),
                        "source_invalid"
                        if source_invalid
                        else ("available" if local_path else "discovered"),
                        source_invalid,
                        local_path,
                        json.dumps(doc, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                imported += 1
        return imported

    def _segment_to_chunk(
        self, row: dict[str, Any], distance: float | None = None
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=row["id"],
            text=row["text_content"],
            source=row["source"],
            title=row["title"],
            link_original=row["link_original"],
            page=row["page"],
            article_number=row["article_number"],
            law_status=row["law_status"],
            distance=distance,
            source_scope=row["source_scope"],
            document_id=row["document_id"],
            metadata=row["metadata"] or {},
        )

    def upsert_legal_segments(self, items: list[dict[str, Any]]) -> int:
        self.initialize()
        if not items:
            return 0
        now = utc_now_iso()
        with self.connection() as conn, conn.cursor() as cur:
            for item in items:
                metadata = item["metadata"]
                refs = metadata.get("article_references") or []
                vector = _vector_literal(item.get("embedding"))
                cur.execute(
                    """
                    INSERT INTO legal_segments (
                        id, legal_document_id, source, title, link_original, page, article_number,
                        article_main, article_references, law_status, source_scope, document_id,
                        diploma_slug, legal_branch, topic_route, text_content, metadata,
                        text_search, embedding, created_at, updated_at
                    )
                    VALUES (
                        %s, NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s::jsonb, to_tsvector('portuguese', %s),
                        %s::double precision[],
                        %s, %s
                    )
                    ON CONFLICT (id) DO UPDATE
                    SET source = EXCLUDED.source,
                        title = EXCLUDED.title,
                        link_original = EXCLUDED.link_original,
                        page = EXCLUDED.page,
                        article_number = EXCLUDED.article_number,
                        article_main = EXCLUDED.article_main,
                        article_references = EXCLUDED.article_references,
                        law_status = EXCLUDED.law_status,
                        source_scope = EXCLUDED.source_scope,
                        document_id = EXCLUDED.document_id,
                        diploma_slug = EXCLUDED.diploma_slug,
                        legal_branch = EXCLUDED.legal_branch,
                        topic_route = EXCLUDED.topic_route,
                        text_content = EXCLUDED.text_content,
                        metadata = EXCLUDED.metadata,
                        text_search = EXCLUDED.text_search,
                        embedding = EXCLUDED.embedding,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        item["id"],
                        metadata.get("source", "Desconhecido"),
                        metadata.get("title", metadata.get("source", "Documento")),
                        metadata.get("link_original"),
                        metadata.get("page"),
                        metadata.get("article_number"),
                        metadata.get("article_main"),
                        refs,
                        metadata.get("law_status", "Nao verificado"),
                        metadata.get("source_scope", "official"),
                        metadata.get("document_id"),
                        metadata.get("diploma_slug"),
                        metadata.get("legal_branch"),
                        metadata.get("topic_route"),
                        item["text"],
                        json.dumps(metadata, ensure_ascii=False),
                        item["text"],
                        vector,
                        now,
                        now,
                    ),
                )
        return len(items)

    def upsert_jurisprudence_cases(self, items: list[dict[str, Any]]) -> int:
        self.initialize()
        if not items:
            return 0
        now = utc_now_iso()
        with self.connection() as conn, conn.cursor() as cur:
            for item in items:
                cur.execute(
                    """
                    INSERT INTO jurisprudence_cases (
                        id, court, chamber, case_number, title, decision_date,
                        publication_date, url, pdf_url, legal_branch, topic_route,
                        summary, metadata, created_at, updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s::jsonb, %s, %s
                    )
                    ON CONFLICT (id) DO UPDATE
                    SET chamber = EXCLUDED.chamber,
                        case_number = EXCLUDED.case_number,
                        title = EXCLUDED.title,
                        decision_date = EXCLUDED.decision_date,
                        publication_date = EXCLUDED.publication_date,
                        url = EXCLUDED.url,
                        pdf_url = EXCLUDED.pdf_url,
                        legal_branch = EXCLUDED.legal_branch,
                        topic_route = EXCLUDED.topic_route,
                        summary = EXCLUDED.summary,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        item["id"],
                        item["court"],
                        item.get("chamber"),
                        item.get("case_number"),
                        item["title"],
                        item.get("decision_date"),
                        item.get("publication_date"),
                        item["url"],
                        item.get("pdf_url"),
                        item.get("legal_branch"),
                        item.get("topic_route"),
                        item.get("summary"),
                        json.dumps(item.get("metadata") or {}, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
        return len(items)

    def available_diploma_slugs(self) -> set[str]:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT diploma_slug FROM legal_segments WHERE diploma_slug IS NOT NULL"
            )
            return {
                row["diploma_slug"] for row in cur.fetchall() if row.get("diploma_slug")
            }

    async def query_legal_segments(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        import asyncio as _asyncio

        self.initialize()
        from app.services.rag.embeddings import embedding_service

        cache_key = query.strip().casefold()
        cached_emb = _embedding_cache.get(cache_key)
        if cached_emb is not None:
            query_vec = cached_emb
        else:
            query_embedding = await embedding_service.embed_query(query)
            query_vec = list(query_embedding) if query_embedding else None
            if query_vec:
                _embedding_cache[cache_key] = query_vec
        query_dim = len(query_vec) if query_vec else 0
        sql = """
            SELECT id, source, title, link_original, page, article_number, law_status,
                   source_scope, document_id, metadata, text_content, embedding,
                   COALESCE(ts_rank(text_search, websearch_to_tsquery('portuguese', %s)), 0) AS lexical_rank
            FROM legal_segments
        """
        clauses: list[str] = []
        params: list[Any] = [query]

        # FTS soft filter: extract OR-terms to narrow scan via GIN index
        # Uses OR logic (match ANY term) — broad enough to preserve recall
        # while reducing the full-table scan from 2000+ rows to ~50-200.
        _fts_or_query = _build_fts_or_query(query)
        if _fts_or_query:
            clauses.append("text_search @@ to_tsquery('portuguese', %s)")
            params.append(_fts_or_query)

        if query_dim:
            clauses.append("embedding IS NOT NULL")
            clauses.append("array_length(embedding, 1) = %s")
            params.append(query_dim)
        if where:
            for key, value in where.items():
                if key.startswith("metadata__"):
                    meta_key = key.split("__", 1)[1]
                    clauses.append("metadata ->> %s = %s")
                    params.extend([meta_key, str(value)])
                else:
                    clauses.append(f"{key} = %s")
                    params.append(value)
        sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY lexical_rank DESC LIMIT %s"
        params.append(max(1, k * 6))

        return await _asyncio.to_thread(
            self._query_legal_segments_sync, query_vec, k, sql, params
        )

    def _query_legal_segments_sync(
        self,
        query_vec: list[float] | None,
        k: int,
        sql: str,
        params: list[Any],
    ) -> list[RetrievedChunk]:
        import numpy as np

        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        scored: list[tuple[float, float, dict[str, Any]]] = []
        query_np = (
            np.array(query_vec, dtype=np.float32) if query_vec is not None else None
        )
        query_norm = float(np.linalg.norm(query_np)) if query_np is not None else 0.0

        for row in rows:
            emb = row.get("embedding")
            if query_np is not None and emb is not None and query_norm > 0:
                emb_list = list(emb) if not isinstance(emb, list) else emb
                emb_arr = np.array(emb_list, dtype=np.float32)
                emb_norm = float(np.linalg.norm(emb_arr))
                if emb_norm > 0:
                    sim = float(np.dot(query_np, emb_arr)) / (query_norm * emb_norm)
                    distance = 1.0 - sim
                else:
                    distance = 1.0
            else:
                distance = 1.0
            lexical = float(row["lexical_rank"] or 0)
            scored.append((distance, -lexical, row))

        scored.sort(key=lambda x: (x[0], x[1]))
        chunks: list[RetrievedChunk] = []
        for distance, _lexical_neg, row in scored[: max(1, k)]:
            chunks.append(self._segment_to_chunk(row, distance=float(distance)))
        return chunks

    def get_branch_prototypes(self) -> dict[str, Any]:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT legal_branch, embedding
                FROM legal_segments
                WHERE legal_branch IS NOT NULL AND embedding IS NOT NULL
                """
            )
            rows = cur.fetchall()
        branch_embeddings: dict[str, list[list[float]]] = {}
        for row in rows:
            branch = row["legal_branch"] or "indeterminado"
            emb = row["embedding"]
            if emb is None:
                continue
            branch_embeddings.setdefault(branch, []).append(list(emb))
        import numpy as np

        prototypes: dict[str, Any] = {}
        for branch, embs in branch_embeddings.items():
            if embs:
                prototypes[branch] = np.mean(np.array(embs, dtype=np.float32), axis=0)
        return prototypes

    def get_document_chunks(
        self, document_id: str, limit: int = 8
    ) -> list[RetrievedChunk]:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, source, title, link_original, page, article_number, law_status,
                       source_scope, document_id, metadata, text_content
                FROM legal_segments
                WHERE document_id = %s
                ORDER BY page ASC NULLS LAST, id ASC
                LIMIT %s
                """,
                (document_id, limit),
            )
            return [self._segment_to_chunk(row) for row in cur.fetchall()]

    def delete_document_chunks(self, document_id: str) -> None:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM legal_segments WHERE document_id = %s", (document_id,)
            )

    def delete_segments_by_metadata(self, **metadata_filters: Any) -> int:
        self.initialize()
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in metadata_filters.items():
            clauses.append(f"{key} = %s")
            params.append(value)
        sql = "DELETE FROM legal_segments"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    def count_segments_by_metadata(self, **metadata_filters: Any) -> int:
        self.initialize()
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in metadata_filters.items():
            clauses.append(f"{key} = %s")
            params.append(value)
        sql = "SELECT COUNT(*) AS total FROM legal_segments"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return int(row["total"] or 0)

    def find_article_chunks(
        self,
        diploma_slug: str,
        article_number: str,
        expected_page: int | None = None,
        limit: int = 8,
    ) -> list[RetrievedChunk]:
        self.initialize()
        normalized_target = str(article_number).replace(".", "").strip()
        if not diploma_slug or not normalized_target:
            return []
        sql = """
            SELECT id, source, title, link_original, page, article_number, law_status,
                   source_scope, document_id, metadata, text_content
            FROM legal_segments
            WHERE diploma_slug = %s
              AND (
                  article_main = %s
                  OR article_number LIKE %s
              )
            ORDER BY page ASC NULLS LAST, id ASC
            LIMIT %s
        """
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    diploma_slug,
                    normalized_target,
                    f"%{normalized_target}%",
                    limit,
                ),
            )
            return [self._segment_to_chunk(row) for row in cur.fetchall()]

    def get_legal_segment_stats(self) -> dict[str, Any]:
        self.initialize()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS total, MAX(updated_at) AS last_update FROM legal_segments"
            )
            row = cur.fetchone() or {"total": 0, "last_update": None}
            return {"total": int(row["total"] or 0), "last_update": row["last_update"]}


postgres_manager = PostgresManager()
