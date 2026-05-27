from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.auth import validate_login
from app.core.logger import configure_logging
from app.services.rag.pipeline import rag_pipeline


@dataclass(slots=True)
class BenchmarkCase:
    persona: str
    question: str
    expected_mode: str
    min_score: float
    must_include_issue: str | None = None


BENCHMARK_CASES = [
    BenchmarkCase(
        persona="leigo",
        question="Fui despedido sem aviso, quais são os meus direitos?",
        expected_mode="grounded",
        min_score=0.75,
    ),
    BenchmarkCase(
        persona="advogado",
        question="Quando um empregador deixa de pagar valores ao trabalhador, como distinguir conflito laboral de relevância penal?",
        expected_mode="limited",
        min_score=0.45,
        must_include_issue="penal_relevance_gap",
    ),
    BenchmarkCase(
        persona="jurista",
        question="Qual é o prazo de recurso no CPP para prisão preventiva?",
        expected_mode="limited",
        min_score=0.40,
        must_include_issue="processual_specificity_gap",
    ),
    BenchmarkCase(
        persona="penal",
        question="Qual é a pena para burla?",
        expected_mode="grounded",
        min_score=0.70,
    ),
    BenchmarkCase(
        persona="meta",
        question="Se a resposta depender de lei processual e lei substantiva de ramos diferentes, como o sistema lida com mistura de fontes e que alertas devolve?",
        expected_mode="limited",
        min_score=0.45,
    ),
    BenchmarkCase(
        persona="juiz",
        question="Como funciona herança em Angola?",
        expected_mode="grounded",
        min_score=0.70,
    ),
    BenchmarkCase(
        persona="leigo-bi",
        question="Qual é o custo da segunda via do bilhete de identidade em Angola?",
        expected_mode="grounded",
        min_score=0.70,
    ),
    BenchmarkCase(
        persona="advogado-sociedades",
        question="Quais são os direitos essenciais dos sócios minoritários numa sociedade por quotas em Angola?",
        expected_mode="grounded",
        min_score=0.68,
    ),
    BenchmarkCase(
        persona="jurista-contencioso",
        question="Qual é o prazo para impugnar judicialmente um acto administrativo em Angola?",
        expected_mode="grounded",
        min_score=0.68,
    ),
    BenchmarkCase(
        persona="familia-divorcio",
        question="Como funciona o divórcio e a regulação familiar básica no Código da Família em Angola?",
        expected_mode="grounded",
        min_score=0.75,
    ),
]


async def run_benchmark(provider: str | None = None) -> dict:
    user = validate_login("admin", "Admin123@")
    results: list[dict] = []
    passed = 0

    for case in BENCHMARK_CASES:
        response = await rag_pipeline.answer_query(
            case.question,
            provider=provider,
            conversation_history=[],
            chat_id=None,
            active_document_id=None,
            user_id=user["id"],
        )
        score = float((response.confidence or {}).get("score") or 0.0)
        issue_codes = [issue.get("code") for issue in (response.validation_issues or [])]
        mode_ok = response.answer_mode == case.expected_mode
        score_ok = score >= case.min_score
        issue_ok = True if not case.must_include_issue else case.must_include_issue in issue_codes
        case_passed = mode_ok and score_ok and issue_ok
        if case_passed:
            passed += 1

        results.append(
            {
                "case": asdict(case),
                "result": {
                    "answer_mode": response.answer_mode,
                    "confidence": response.confidence,
                    "validation_issues": response.validation_issues,
                    "classification": response.classification,
                    "legal_basis": response.legal_basis,
                    "sources": [source.model_dump() for source in response.sources],
                    "answer_preview": response.answer[:800],
                },
                "checks": {
                    "mode_ok": mode_ok,
                    "score_ok": score_ok,
                    "issue_ok": issue_ok,
                    "passed": case_passed,
                },
            }
        )

    total = len(BENCHMARK_CASES)
    ratio = passed / total if total else 0.0
    score_10 = round(ratio * 10, 2)
    return {
        "summary": {
            "total_cases": total,
            "passed_cases": passed,
            "pass_ratio": round(ratio, 4),
            "score_0_to_10": score_10,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--output", default="C:/Projectos/TCC/backend/data/processed/legal_benchmark_results.json")
    args = parser.parse_args()

    configure_logging()
    payload = asyncio.run(run_benchmark(provider=args.provider))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output_path))
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
