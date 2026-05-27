"""Força piramyd antes de importar o módulo de config que faz load_dotenv(override=True)."""

import os

os.environ["DEFAULT_LLM_PROVIDER"] = "piramyd"
os.environ["OPENAI_API_KEY"] = ""  # limpa para evitar fallback acidental

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import asyncio
import time
from dataclasses import asdict, dataclass
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

QUESTIONS_COMPLEXAS = [
    {
        "id": 1,
        "category": "Contratos",
        "question": "Um contrato de compra e venda de imóveis em Angola pode ser resolvido por incumprimento contractual? Quais são os fundamentos legais e prazos aplicáveis?",
        "complexity": "Alta",
    },
    {
        "id": 2,
        "category": "Trabalho",
        "question": "Um trabalhador dispensado sem justa causa em Angola tem direito a indemnização? Como se calcula a mesma e quais os prazos para contestação?",
        "complexity": "Alta",
    },
    {
        "id": 3,
        "category": "Direito Penal",
        "question": "Em que circunstâncias se configura o crime de burla qualificada em Angola e qual a pena aplicável? Existe possibilidade de suspensão da pena?",
        "complexity": "Alta",
    },
    {
        "id": 4,
        "category": "Sucessões",
        "question": "Como se processa a partilha de herança quando existem herdeiros menores em Angola? Qual a intervenção do tribunal e dos pais/encarregados de educação?",
        "complexity": "Alta",
    },
    {
        "id": 5,
        "category": "Arrendamento",
        "question": "Um senhorio pode não renovar um contrato de arrendamento habitacional em Angola sem invocar justo motivo? O que diz a lei sobre o direito de preferência do inquilino?",
        "complexity": "Alta",
    },
    {
        "id": 6,
        "category": "Responsabilidade Civil",
        "question": "Uma empresa que comercializa produtos defeituosos responde civilmente pelos danos causados ao consumidor em Angola? Quais os pressupostos da responsabilidade do produtor?",
        "complexity": "Alta",
    },
    {
        "id": 7,
        "category": "Processo Civil",
        "question": "O que acontece se uma das partes num processo civil em Angola não comparece à audiência de julgamento? O julgamento pode prosseguir à revelia?",
        "complexity": "Média",
    },
    {
        "id": 8,
        "category": "Direito Societário",
        "question": "Um sócio minoritário de uma sociedade anónima em Angola pode impugnar uma deliberação da assembleia geral que considera lesiva dos seus direitos? Em que prazo e perante que órgão?",
        "complexity": "Alta",
    },
    {
        "id": 9,
        "category": "Direito Fiscal",
        "question": "Uma empresa angolana que deixou de pagar impostos durante 3 anos pode ainda ser autuada pela Autoridade Tributária? Qual o prazo prescricional aplicável?",
        "complexity": "Alta",
    },
    {
        "id": 10,
        "category": "Direito da Família",
        "question": "Em caso de divórcio litigioso em Angola, como se determina a guarda dos filhos menores? Que peso tem a vontade da criança na decisão do tribunal?",
        "complexity": "Alta",
    },
]


async def run_question(q: dict) -> dict:
    result = {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "complexity": q["complexity"],
    }
    try:
        start = time.time()
        user = validate_login("admin", "Admin123@")
        response = await rag_pipeline.answer_query(
            q["question"],
            provider="piramyd",
            conversation_history=[],
            chat_id=None,
            active_document_id=None,
            user_id=user["id"],
        )
        elapsed = round(time.time() - start, 2)
        result.update(
            {
                "answer_mode": response.answer_mode,
                "confidence": response.confidence,
                "provider_used": response.provider_used,
                "latency_seconds": elapsed,
                "sources": [
                    {
                        "title": s.title,
                        "source": s.source,
                        "page": s.page,
                        "article": s.article_number,
                    }
                    for s in response.sources
                ],
                "validation_issues": response.validation_issues,
                "legal_basis": response.legal_basis,
                "answer_preview": response.answer[:500] if response.answer else None,
                "error": None,
            }
        )
    except Exception as exc:
        result.update(
            {"error": str(exc), "latency_seconds": round(time.time() - start, 2)}
        )
    return result


async def main():
    configure_logging()
    print("=" * 90)
    print("BENCHMARK COMPLETO - SISTEMA DE ASSISTÊNCIA JURÍDICA ANGOLANA (via Piramyd)")
    print("=" * 90)

    # --- PARTE 1: Benchmark legal (casos com expectativas) ---
    print("\n>>> PARTE 1: Benchmark Legal (casos com validação de modo/score) <<<\n")
    user = validate_login("admin", "Admin123@")
    results_legal = []
    passed = 0

    for case in BENCHMARK_CASES:
        print(f"[{case.persona}] {case.question[:80]}...")
        resp = await rag_pipeline.answer_query(
            case.question,
            provider="piramyd",
            conversation_history=[],
            chat_id=None,
            active_document_id=None,
            user_id=user["id"],
        )
        score = float((resp.confidence or {}).get("score") or 0.0)
        issue_codes = [i.get("code") for i in (resp.validation_issues or [])]
        mode_ok = resp.answer_mode == case.expected_mode
        score_ok = score >= case.min_score
        issue_ok = (case.must_include_issue is None) or (
            case.must_include_issue in issue_codes
        )
        case_pass = mode_ok and score_ok and issue_ok
        if case_pass:
            passed += 1
        print(
            f"  -> mode={resp.answer_mode} (esperado={case.expected_mode}) score={score} (min={case.min_score}) issues={issue_codes} | {'PASS' if case_pass else 'FAIL'}"
        )
        results_legal.append(
            {
                "persona": case.persona,
                "question": case.question,
                "answer_mode": resp.answer_mode,
                "score": score,
                "issues": issue_codes,
                "passed": case_pass,
            }
        )

    total = len(BENCHMARK_CASES)
    print(
        f"\nLegal Benchmark: {passed}/{total} passed | Score: {round(passed / total * 10, 2)}/10"
    )

    # --- PARTE 2: Perguntas complexas ---
    print("\n>>> PARTE 2: Perguntas Complexas (10 áreas do direito angolano) <<<\n")
    results_complex = []
    for q in QUESTIONS_COMPLEXAS:
        print(f"[{q['id']}/10] {q['category']} ({q['complexity']})...")
        r = await run_question(q)
        results_complex.append(r)
        print(
            f"  -> mode={r.get('answer_mode')} | conf={r.get('confidence', {}).get('score') if r.get('confidence') else 'N/A'} | latency={r.get('latency_seconds')}s | provider={r.get('provider_used')} | err={r.get('error')}"
        )

    # --- RESUMO ---
    print("\n" + "=" * 90)
    print("RESUMO DO BENCHMARK")
    print("=" * 90)

    # Legal benchmark summary
    print(f"\n--- Legal Benchmark (casos com validação) ---")
    print(
        f"  Total: {total} | Passed: {passed} | Score: {round(passed / total * 10, 2)}/10"
    )

    # Complex questions summary
    successful = [r for r in results_complex if not r.get("error")]
    failed = [r for r in results_complex if r.get("error")]
    avg_latency = sum(
        r["latency_seconds"] for r in results_complex if r.get("latency_seconds")
    ) / max(len(results_complex), 1)
    scores = [
        r.get("confidence", {}).get("score") or 0
        for r in results_complex
        if r.get("confidence")
    ]
    avg_conf = sum(scores) / max(len(scores), 1)

    print(f"\n--- Perguntas Complexas (10 áreas) ---")
    print(f"  Sucesso: {len(successful)}/{len(QUESTIONS_COMPLEXAS)}")
    print(f"  Falhas: {len(failed)}")
    print(f"  Latência média: {avg_latency:.2f}s")
    print(f"  Confiança média: {avg_conf:.2f}")
    print(f"  Provider usado: piramyd")

    modes = {}
    for r in results_complex:
        m = r.get("answer_mode", "N/A")
        modes[m] = modes.get(m, 0) + 1
    print(f"  Modos de resposta: {modes}")

    print(f"\n--- Detalhe por Pergunta ---")
    for r in results_complex:
        print(f"\n[{r['id']}] {r['category']} - {r['complexity']}")
        print(f"  Q: {r['question'][:100]}")
        if r.get("error"):
            print(f"  ERRO: {r['error']}")
        else:
            print(
                f"  Modo: {r.get('answer_mode')} | Confiança: {r.get('confidence', {}).get('level')} ({r.get('confidence', {}).get('score')})"
            )
            print(
                f"  Latência: {r.get('latency_seconds')}s | Provider: {r.get('provider_used')}"
            )
            print(f"  Fontes: {len(r.get('sources') or [])}")
            if r.get("validation_issues"):
                print(f"  Issues: {[i.get('code') for i in r['validation_issues']]}")
            if r.get("answer_preview"):
                print(f"  Resposta: {r['answer_preview'][:300]}...")

    # --- OUTPUT JSON ---
    output = {
        "benchmark_legal": {
            "total": total,
            "passed": passed,
            "score_0_10": round(passed / total * 10, 2),
            "cases": results_legal,
        },
        "perguntas_complexas": {
            "total": len(QUESTIONS_COMPLEXAS),
            "sucesso": len(successful),
            "falhas": len(failed),
            "latencia_media": round(avg_latency, 2),
            "confianca_media": round(avg_conf, 2),
            "modos": modes,
            "detalhes": results_complex,
        },
    }
    output_path = Path(
        "C:/Projectos/TCC/backend/data/processed/benchmark_completo.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResultados salvos em: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
