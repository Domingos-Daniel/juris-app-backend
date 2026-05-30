from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Pergunta do utilizador")
    provider: str | None = Field(
        default=None,
        description="Override opcional do provedor LLM: deepseek, openrouter, openai",
    )
    conversation_history: list[str] = Field(
        default_factory=list,
        description="Historico textual curto da conversa para manter contexto do fio",
    )
    chat_id: str | None = Field(
        default=None, description="Identificador do chat persistido"
    )
    active_document_id: str | None = Field(
        default=None, description="Documento PDF ativo para contexto prioritario"
    )


class SourceItem(BaseModel):
    title: str
    source: str
    link_original: str | None = None
    deep_link: str | None = None
    page: int | None = None
    article_number: str | None = None
    law_status: str = "Nao verificado"
    excerpt: str | None = None
    source_scope: str = "official"
    source_kind: str | None = None
    document_id: str | None = None
    attribution_text: str | None = Field(
        default=None, description="Trecho do chunk recuperado que contém este artigo"
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    provider_used: str
    chat_id: str
    active_document_id: str | None = None
    answer_mode: str = "limited"
    confidence: dict[str, Any] | None = None
    classification: dict[str, Any] | None = None
    legal_basis: list[dict[str, Any]] = Field(default_factory=list)
    validation_issues: list[dict[str, Any]] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    verified_articles: list[dict[str, Any]] = Field(default_factory=list)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user: dict


class DocumentIngestResponse(BaseModel):
    filename: str
    chunks_created: int
    used_ocr: bool
    ocr_pages: int = 0


class IngestSummary(BaseModel):
    processed_files: list[DocumentIngestResponse]
    total_chunks: int


class UserDocumentItem(BaseModel):
    id: str
    filename: str
    display_name: str
    storage_path: str | None = None
    mime_type: str
    size_bytes: int
    status: str
    created_at: str
    page_count: int = 0
    chunks_created: int = 0
    extraction_mode: str = "direct"
    quality_status: str = "good"
    summary: str | None = None
    preview_text: str | None = None
    category: str | None = None
    usage_count: int = 0
    last_used_at: str | None = None


class UserDocumentListResponse(BaseModel):
    items: list[UserDocumentItem]


class DocumentRenameRequest(BaseModel):
    display_name: str = Field(..., min_length=2, max_length=180)


class DocumentUseRequest(BaseModel):
    chat_id: str | None = None


class ChatMessageItem(BaseModel):
    id: str
    role: str
    content: str
    provider_used: str | None = None
    created_at: str
    sources: list[SourceItem] = Field(default_factory=list)


class ChatListItem(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    active_document_id: str | None = None
    messages: list[ChatMessageItem] = Field(default_factory=list)


class ChatListResponse(BaseModel):
    items: list[ChatListItem]


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    title: str
    link_original: str | None
    page: int | None
    article_number: str | None
    law_status: str
    distance: float | None = None
    source_scope: str = "official"
    document_id: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class QueryRecord:
    question: str
    answer: str
    timestamp: datetime
