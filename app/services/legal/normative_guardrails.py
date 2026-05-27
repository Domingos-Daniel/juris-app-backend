from __future__ import annotations

from app.services.legal.models import (
    LegalClassification,
    RetrievalResult,
    ValidatedLegalBasisItem,
    ValidationIssue,
)

VIGENCY_QUERY_MARKERS = (
    "vigência",
    "vigencia",
    "revogado",
    "revogada",
    "alterado",
    "alterada",
    "em vigor",
    "continua em vigor",
    "ainda vale",
    "ainda vigora",
)
CONFLICT_QUERY_MARKERS = (
    "conflito normativo",
    "qual prevalece",
    "norma especial",
    "revoga",
    "revogado",
    "revogada",
    "altera",
    "alterado",
    "alterada",
)
CONFLICT_TEXT_MARKERS = (
    "revoga",
    "revogado",
    "revogada",
    "altera",
    "altera a",
    "repristina",
    "derroga",
    "conflito",
)


def _normalize(text: str | None) -> str:
    return (text or "").strip().casefold()


def _issue(
    code: str, message: str, severity: str = "medium"
) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, severity=severity)  # type: ignore[arg-type]


def _requested_vigency_check(classification: LegalClassification) -> bool:
    haystack = _normalize(classification.query_text)
    return any(marker in haystack for marker in VIGENCY_QUERY_MARKERS)


def _requested_conflict_check(classification: LegalClassification) -> bool:
    haystack = _normalize(classification.query_text)
    return any(marker in haystack for marker in CONFLICT_QUERY_MARKERS)


def _document_kind(chunk) -> str:
    return str((chunk.metadata or {}).get("document_kind") or "legislation")


def _diploma_slug(chunk) -> str:
    return str((chunk.metadata or {}).get("diploma_slug") or "")


def _law_status(chunk) -> str:
    return str((chunk.metadata or {}).get("law_status") or chunk.law_status or "")


def _is_jurisprudence(chunk) -> bool:
    return _document_kind(chunk) == "jurisprudence"


class NormativeGuardrailsService:
    def analyze(
        self,
        classification: LegalClassification,
        retrieval: RetrievalResult,
        confirmed: list[ValidatedLegalBasisItem],
    ) -> tuple[str, list[str], list[ValidationIssue], list[ValidatedLegalBasisItem]]:
        official = retrieval.official_evidence
        normative = [item for item in official if not _is_jurisprudence(item.chunk)]
        jurisprudence = [item for item in official if _is_jurisprudence(item.chunk)]
        issues: list[ValidationIssue] = []
        notes: list[str] = []
        asks_vigency = _requested_vigency_check(classification)
        asks_conflict = _requested_conflict_check(classification)

        statuses = {
            _law_status(item.chunk)
            for item in normative
            if _law_status(item.chunk).strip()
        }
        normalized_statuses = {_normalize(status) for status in statuses}

        normative_status = "unverified"
        if any("em vigor" in status for status in normalized_statuses):
            normative_status = "confirmed_in_force"
        elif statuses:
            normative_status = "partially_known"

        if asks_vigency and normative_status != "confirmed_in_force":
            issues.append(
                _issue(
                    "vigency_unverified",
                    "A pergunta exige confirmação de vigência, mas o contexto recuperado ainda não confirma claramente se a norma está em vigor.",
                    "high",
                )
            )

        requested_slugs = {
            str((item.chunk.metadata or {}).get("diploma_slug") or "")
            for item in official[:4]
            if (item.chunk.metadata or {}).get("diploma_slug")
        }
        if (
            (asks_vigency or asks_conflict)
            and classification.requested_diplomas
            and len(requested_slugs) >= 2
            and not classification.needs_multi_branch_handling
        ):
            issues.append(
                _issue(
                    "normative_conflict",
                    "Foram recuperados diplomas principais concorrentes para a mesma resposta; é preciso separar qual norma domina o caso.",
                    "high",
                )
            )

        if (asks_vigency or asks_conflict) and any(
            any(marker in _normalize(item.chunk.text) for marker in CONFLICT_TEXT_MARKERS)
            for item in normative[:4]
        ) and not confirmed:
            issues.append(
                _issue(
                    "citator_gap",
                    "O contexto sugere alteração, revogação ou conflito normativo, mas a resposta ainda não tem confirmação jurídica suficiente sobre qual regime prevalece.",
                    "high",
                )
            )

        if jurisprudence:
            notes.append(
                f"{len(jurisprudence)} fonte(s) jurisprudenciais recuperadas como apoio interpretativo."
            )

        jurisprudence_basis = [
            ValidatedLegalBasisItem(
                diploma=item.chunk.title,
                article=None,
                page=item.chunk.page,
                source_scope=item.chunk.source_scope,
                confirmed=False,
                excerpt=item.chunk.text[:420],
                deep_link=item.chunk.link_original,
            )
            for item in jurisprudence[:2]
        ]

        return normative_status, notes, issues, jurisprudence_basis


normative_guardrails_service = NormativeGuardrailsService()
