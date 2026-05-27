from __future__ import annotations

from app.services.legal.article_verifier import VerifiedArticle
from app.services.legal.models import (
    ConfidenceResult,
    LegalClassification,
    RetrievalResult,
    ValidationResult,
)


class LegalConfidenceService:
    def score(
        self,
        classification: LegalClassification,
        retrieval: RetrievalResult,
        validation: ValidationResult,
        verified_articles: list[VerifiedArticle] | None = None,
    ) -> ConfidenceResult:
        verified = verified_articles or []
        verified_count = sum(1 for va in verified if va.status != "not_found")
        cited_count = sum(
            len(item.article or "") > 0
            for item in validation.confirmed_legal_basis
            + validation.prudential_legal_basis
            if item.article
        )
        total_cited = cited_count or len(verified) or 1
        issue_codes = {issue.code for issue in validation.issues}

        score = 0.10

        if retrieval.official_evidence:
            score += 0.12
        if validation.confirmed_legal_basis:
            score += 0.12

        verification_ratio = min(verified_count / max(total_cited, 1), 1.0)
        score += verification_ratio * 0.46

        if validation.answer_mode == "refused":
            score = min(score, 0.40)
        elif validation.answer_mode == "limited":
            score = min(score, 0.50)
        elif verification_ratio < 0.3:
            score = min(score, 0.60)

        if not verification_ratio:
            score -= 0.10

        if validation.unsupported_articles:
            score -= 0.30

        if classification.main_branch not in ("indeterminado", "misto"):
            from app.services.legal.retrieval import _chunk_branch

            off_branch = sum(
                1
                for ev in retrieval.official_evidence
                if _chunk_branch(ev.chunk) != classification.main_branch
            )
            mismatch_ratio = off_branch / max(len(retrieval.official_evidence), 1)
            if mismatch_ratio > 0.3:
                score -= round(mismatch_ratio * 0.18, 3)

        for pen in 0.08, 0.12:
            if "penal_relevance_gap" in issue_codes:
                score -= pen
            if "processual_specificity_gap" in issue_codes:
                score -= pen
        if (
            "strict_corpus_missing" in issue_codes
            or "strict_corpus_mismatch" in issue_codes
        ):
            score -= 0.10
        if (
            "strict_confirmation_gap" in issue_codes
            or "weak_article_confirmation" in issue_codes
        ):
            score -= 0.10
        if "branch_gap" in issue_codes:
            score -= 0.15
        if "branch_mismatch" in issue_codes:
            score -= 0.18
        if "no_official_support" in issue_codes:
            score -= 0.20
        if "followup_anchor_unresolved" in issue_codes:
            score -= 0.22
        if "normative_conflict" in issue_codes or "citator_gap" in issue_codes:
            score -= 0.18
        if "vigency_unverified" in issue_codes:
            score -= 0.15
        if "unsupported_article" in issue_codes or "unverified_article" in issue_codes:
            score -= 0.05
        if validation.source_cross_contamination:
            score -= 0.05
        if validation.missing_branches:
            score -= 0.08

        if not retrieval.official_evidence:
            score = min(score, 0.45)
        if (
            classification.requires_strict_corpus_match
            and not validation.sufficient_legal_support
        ):
            score = min(score, 0.55)
        if any(
            code in issue_codes
            for code in (
                "followup_anchor_unresolved",
                "normative_conflict",
                "citator_gap",
                "vigency_unverified",
            )
        ):
            score = min(score, 0.45)
        if (
            validation.answer_mode == "grounded_with_caveat"
            and retrieval.official_evidence
        ):
            score = min(max(score, 0.35), 0.55)
        if (
            validation.answer_mode == "grounded"
            and len(retrieval.official_evidence) <= 2
        ):
            score = max(score, 0.40)
        if validation.answer_mode in {"limited", "grounded_with_caveat"}:
            score = min(score, 0.55)

        if (
            verification_ratio > 0.8
            and validation.sufficient_legal_support
            and validation.answer_mode == "grounded"
        ):
            score = max(score, 0.70)
        elif verification_ratio > 0.4 and validation.answer_mode == "grounded":
            score = max(score, 0.45)

        score = max(0.0, min(1.0, score))
        if score >= 0.70:
            level = "alta"
        elif score >= 0.40:
            level = "media"
        else:
            level = "baixa"

        reasons: list[str] = []
        if verification_ratio > 0:
            reasons.append(
                f"{verified_count} de {total_cited} artigos confirmados no corpus."
            )
        if verification_ratio == 0 and total_cited > 0:
            reasons.append("Nenhum artigo citado foi confirmado no corpus indexado.")
        if verification_ratio < 0.5:
            reasons.append(
                "Confiança limitada — menos de metade dos artigos verificados."
            )
        if validation.answer_mode == "limited":
            reasons.append(
                "Modo limitado — o sistema não garante total precisão normativa."
            )
        if not reasons:
            reasons.append(
                "Equilíbrio entre suporte recuperado, verificação e limites ainda abertos."
            )

        return ConfidenceResult(
            level=level,
            score=round(score, 3),
            reasons=reasons[:5],
        )


legal_confidence_service = LegalConfidenceService()
