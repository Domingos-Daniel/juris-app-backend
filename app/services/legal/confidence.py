"""Legal confidence scoring — meaningful 0-1 score based on retrieval quality."""

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
        total_cited = cited_count or max(verified_count, 1)
        verification_ratio = min(verified_count / max(total_cited, 1), 1.0)
        issue_codes = {issue.code for issue in validation.issues}
        evidence_count = len(retrieval.official_evidence)

        # --- Base score from retrieval presence ---
        score = 0.30 if evidence_count > 0 else 0.10

        # --- Retrieval quality (distance-based) ---
        if evidence_count > 0:
            distances = [
                ev.chunk.distance
                for ev in retrieval.official_evidence[:6]
                if ev.chunk.distance is not None
            ]
            if distances:
                avg_distance = sum(distances) / len(distances)
                # cosine distance: 0.0 = identical, 0.2 = very close, 1.0 = orthogonal
                if avg_distance < 0.15:
                    score += 0.35
                elif avg_distance < 0.25:
                    score += 0.25
                elif avg_distance < 0.40:
                    score += 0.15
                else:
                    score += 0.05

        # --- Evidence count ---
        if evidence_count >= 6:
            score += 0.10
        elif evidence_count >= 3:
            score += 0.06

        # --- Legal basis confirmed ---
        if validation.confirmed_legal_basis:
            score += 0.10

        # --- Verification bonus (if articles were verified) ---
        if verification_ratio > 0:
            score += verification_ratio * 0.12

        # --- Branch alignment bonus ---
        if classification.main_branch not in ("indeterminado", "misto"):
            from app.services.legal.retrieval import _chunk_branch

            on_branch = sum(
                1
                for ev in retrieval.official_evidence
                if _chunk_branch(ev.chunk) == classification.main_branch
            )
            if on_branch >= 3:
                score += 0.08
            elif on_branch >= 1:
                score += 0.04

        # --- Penalties ---
        issue_penalties = {
            "no_official_support": 0.25,
            "followup_anchor_unresolved": 0.22,
            "normative_conflict": 0.18,
            "citator_gap": 0.18,
            "branch_mismatch": 0.15,
            "branch_gap": 0.12,
            "penal_relevance_gap": 0.08,
            "processual_specificity_gap": 0.08,
            "strict_corpus_missing": 0.10,
            "strict_corpus_mismatch": 0.10,
        }
        for code, penalty in issue_penalties.items():
            if code in issue_codes:
                score -= penalty

        if validation.unsupported_articles:
            score -= 0.25
        if validation.source_cross_contamination:
            score -= 0.05
        if validation.missing_branches:
            score -= 0.06

        # --- Mode adjustments ---
        if validation.answer_mode == "refused":
            score = min(score, 0.30)
        elif validation.answer_mode == "limited":
            score = min(score, 0.45)
        elif validation.answer_mode == "grounded_with_caveat":
            score = min(score, 0.60)

        # --- Clamp and determine level ---
        score = max(0.05, min(1.0, score))

        if score >= 0.70:
            level = "alta"
        elif score >= 0.45:
            level = "media"
        else:
            level = "baixa"

        # --- Reasons ---
        reasons: list[str] = []
        if evidence_count >= 6:
            reasons.append(f"{evidence_count} fontes legislativas recuperadas.")
        elif evidence_count > 0:
            reasons.append(f"{evidence_count} fonte(s) legislativa(s) recuperada(s).")
        else:
            reasons.append("Nenhuma fonte legislativa encontrada no corpus.")

        if verification_ratio > 0:
            reasons.append(f"{verified_count} de {total_cited} artigos confirmados.")
        elif verified_articles:
            reasons.append("Artigos citados nao foram verificados no corpus.")

        if validation.confirmed_legal_basis:
            reasons.append("Base legal confirmada no contexto recuperado.")

        if (
            classification.main_branch not in ("indeterminado", "misto")
            and evidence_count > 0
        ):
            from app.services.legal.retrieval import _chunk_branch

            on_branch = sum(
                1
                for ev in retrieval.official_evidence
                if _chunk_branch(ev.chunk) == classification.main_branch
            )
            if on_branch >= 3:
                reasons.append(
                    f"Forte alinhamento com o ramo {classification.main_branch}."
                )
            elif on_branch > 0:
                reasons.append(
                    f"Alinhamento parcial com o ramo {classification.main_branch}."
                )

        return ConfidenceResult(
            level=level,
            score=round(score, 3),
            reasons=reasons[:5],
        )


legal_confidence_service = LegalConfidenceService()
