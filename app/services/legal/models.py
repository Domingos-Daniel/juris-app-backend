from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from app.db.models import RetrievedChunk, SourceItem

LegalBranch = Literal[
    "laboral",
    "penal",
    "civil",
    "constitucional",
    "administrativo",
    "comercial",
    "tributario",
    "familia",
    "propriedade",
    "sucessorio",
    "misto",
    "indeterminado",
]
RequestType = Literal[
    "explicacao_simples",
    "analise_tecnica",
    "comparacao",
    "passos_praticos",
    "documentos_prova",
    "competencia_institucional",
    "risco_juridico",
    "estrategia_processual",
    "minuta_documental",
]
SpecificityLevel = Literal[
    "geral",
    "factual",
    "follow_up",
    "comparacao_multi_ramo",
    "validacao_base_legal",
    "meta_sistema",
    "sucessorio",
]
AudienceType = Literal["leigo", "misto", "tecnico"]
ConfidenceLevel = Literal["alta", "media", "baixa"]
AnswerMode = Literal["grounded", "limited", "refused", "grounded_with_caveat"]
TopicRoute = Literal[
    "geral",
    "cpp",
    "cpc",
    "contencioso_admin",
    "processo_administrativo",
    "identificacao_civil",
    "estatuto_magistrados",
    "tribunal_supremo_organica",
    "familia",
    "terras",
    "sociedades",
    "tributario",
    "iva",
    "civil_obrigacoes",
    "sucessoes",
    "laboral",
    "penal_substantivo",
    "constitucional",
    "drafting",
]
NormTypeNeeded = Literal[
    "substantiva", "processual", "misto", "administrativo_operacional"
]
DocumentRole = Literal[
    "codigo_base",
    "lei_especial",
    "estatuto",
    "organica",
    "tributaria",
    "identificacao_civil",
    "codigo_processual",
    "constituicao",
]


class LegalClassification(BaseModel):
    query_text: str = ""
    main_branch: LegalBranch
    branch_candidates: list[LegalBranch] = Field(default_factory=list)
    request_type: RequestType
    specificity: SpecificityLevel
    audience: AudienceType
    is_follow_up: bool = False
    is_correction: bool = False
    is_transformation: bool = False
    semantic_confidence: float = 0.0
    transformation_type: str = "none"
    topic_route: TopicRoute = "geral"
    search_query: str = ""
    norm_type_needed: NormTypeNeeded = "misto"
    requires_strict_corpus_match: bool = False
    drafting_mode: bool = False
    explicit_branch_override: bool = False
    requested_article_numbers: list[str] = Field(default_factory=list)
    requested_diplomas: list[str] = Field(default_factory=list)
    needs_article_validation: bool = False
    needs_source_separation: bool = False
    needs_practical_guidance: bool = False
    needs_multi_branch_handling: bool = False
    conversation_branch_hint: LegalBranch | None = None
    conversation_topic_hint: TopicRoute | None = None
    conversation_norm_type_hint: NormTypeNeeded | None = None
    needs_clarification: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class RetrievalEvidence:
    query_used: str
    chunk: RetrievedChunk
    score: float
    retrieval_reason: str
    source_bucket: str


@dataclass(slots=True)
class BranchEvidenceGroup:
    branch: LegalBranch
    evidences: list[RetrievalEvidence] = field(default_factory=list)
    coverage_gap: bool = False


@dataclass(slots=True)
class RetrievalResult:
    classification: LegalClassification
    official_evidence: list[RetrievalEvidence] = field(default_factory=list)
    user_evidence: list[RetrievalEvidence] = field(default_factory=list)
    branch_groups: list[BranchEvidenceGroup] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    missing_branches: list[LegalBranch] = field(default_factory=list)
    retrieval_notes: list[str] = field(default_factory=list)


class LLMAnswerDraft(BaseModel):
    rich_content: str = ""
    cited_articles: list[str] = Field(default_factory=list)
    cited_diplomas: list[str] = Field(default_factory=list)


class ValidatedLegalBasisItem(BaseModel):
    diploma: str
    article: str | None = None
    page: int | None = None
    source_scope: str = "official"
    confirmed: bool = True
    excerpt: str | None = None
    deep_link: str | None = None


class ValidationIssue(BaseModel):
    code: str
    message: str
    severity: Literal["high", "medium", "low"]


class ValidationResult(BaseModel):
    confirmed_legal_basis: list[ValidatedLegalBasisItem] = Field(default_factory=list)
    prudential_legal_basis: list[ValidatedLegalBasisItem] = Field(default_factory=list)
    jurisprudence_basis: list[ValidatedLegalBasisItem] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    missing_branches: list[LegalBranch] = Field(default_factory=list)
    unsupported_articles: list[str] = Field(default_factory=list)
    source_cross_contamination: bool = False
    sufficient_legal_support: bool = False
    official_sources_used: int = 0
    user_sources_used: int = 0
    answer_mode: AnswerMode = "limited"
    normative_status: str = "unverified"
    normative_notes: list[str] = Field(default_factory=list)


class ConfidenceResult(BaseModel):
    level: ConfidenceLevel
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class LegalAnalysisPayload(BaseModel):
    classification: LegalClassification
    validation: ValidationResult
    confidence: ConfidenceResult


class BenchmarkCaseResult(BaseModel):
    question: str
    answer: str
    confidence: ConfidenceResult
    legal_basis: list[ValidatedLegalBasisItem] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    classification: LegalClassification
    sources: list[SourceItem] = Field(default_factory=list)
