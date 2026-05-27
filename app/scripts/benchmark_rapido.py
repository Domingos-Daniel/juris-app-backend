"""Benchmark rápido: testa 10 perguntas complexas uma a uma com timeout individual."""

import os, sys, json, asyncio, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["DEFAULT_LLM_PROVIDER"] = "piramyd"

import logging

logging.basicConfig(level=logging.WARNING)  # reduz verbosidade

from app.core.auth import validate_login
from app.services.rag.pipeline import rag_pipeline

QUESTIONS = [
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


async def run_one(q, user_id):
    start = time.time()
    try:
        resp = await asyncio.wait_for(
            rag_pipeline.answer_query(
                q["question"],
                provider="piramyd",
                conversation_history=[],
                chat_id=None,
                active_document_id=None,
                user_id=user_id,
            ),
            timeout=120,
        )
        elapsed = round(time.time() - start, 2)
        return {
            "id": q["id"],
            "category": q["category"],
            "complexity": q["complexity"],
            "question": q["question"],
            "answer_mode": resp.answer_mode,
            "confidence": resp.confidence,
            "provider_used": resp.provider_used,
            "latency_seconds": elapsed,
            "num_sources": len(resp.sources),
            "sources": [
                {"title": s.title, "source": s.source, "page": s.page}
                for s in resp.sources[:3]
            ],
            "validation_issues": [
                i.get("code") for i in (resp.validation_issues or [])
            ],
            "legal_basis": [
                {"diploma": b.get("diploma"), "article": b.get("article")}
                for b in (resp.legal_basis or [])
            ],
            "answer_preview": resp.answer[:400] if resp.answer else None,
            "error": None,
        }
    except asyncio.TimeoutError:
        return {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "error": "TIMEOUT (120s)",
            "latency_seconds": round(time.time() - start, 2),
        }
    except Exception as e:
        return {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "error": str(e)[:200],
            "latency_seconds": round(time.time() - start, 2),
        }


async def main():
    print("=" * 90)
    print("BENCHMARK - PERGUNTAS COMPLEXAS SOBRE LEGISLAÇÃO ANGOLANA")
    print("=" * 90)
    user = validate_login("admin", "Admin123@")
    results = []
    for q in QUESTIONS:
        print(f"\n[{q['id']}/10] {q['category']} ({q['complexity']})...")
        sys.stdout.flush()
        r = await run_one(q, user["id"])
        results.append(r)
        status = (
            r.get("error")
            or f"{r.get('answer_mode', '?')} conf={r.get('confidence', {}).get('score', 'N/A') if r.get('confidence') else 'N/A'} t={r.get('latency_seconds', 0)}s src={r.get('num_sources', 0)}"
        )
        print(f"  => {status}")

    # SUMMARY
    print("\n" + "=" * 90)
    print("RESUMO")
    print("=" * 90)
    ok = [r for r in results if not r.get("error")]
    fail = [r for r in results if r.get("error")]
    avg_lat = sum(
        r["latency_seconds"] for r in results if r.get("latency_seconds")
    ) / max(len(results), 1)
    avgs = [
        r.get("confidence", {}).get("score") or 0 for r in ok if r.get("confidence")
    ]
    avg_conf = sum(avgs) / max(len(avgs), 1) if avgs else 0

    print(f"  Total: {len(QUESTIONS)} | Sucesso: {len(ok)} | Falhas: {len(fail)}")
    print(f"  Latência média: {avg_lat:.1f}s")
    print(f"  Confiança média: {avg_conf:.2f}")
    modes = {}
    for r in ok:
        m = r.get("answer_mode", "?")
        modes[m] = modes.get(m, 0) + 1
    print(f"  Modos: {modes}")

    for r in results:
        print(f"\n[{r['id']}] {r.get('category', '?')}")
        print(f"  Q: {r.get('question', '')[:100]}")
        if r.get("error"):
            print(f"  ERRO: {r['error']}")
        else:
            print(
                f"  Modo: {r.get('answer_mode')} | Conf: {r.get('confidence', {}).get('level', '?')} ({r.get('confidence', {}).get('score', '?')})"
            )
            print(
                f"  Lat: {r.get('latency_seconds')}s | Fontes: {r.get('num_sources')} | Prov: {r.get('provider_used')}"
            )
            if r.get("validation_issues"):
                print(f"  Issues: {r['validation_issues']}")
            if r.get("legal_basis"):
                print(f"  Bases legais: {r['legal_basis'][:3]}")
            if r.get("answer_preview"):
                print(f"  Resposta: {r['answer_preview'][:300]}")

    out = Path(
        "C:/Projectos/TCC/backend/data/processed/benchmark_perguntas_complexas.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados salvos em: {out}")


if __name__ == "__main__":
    asyncio.run(main())
