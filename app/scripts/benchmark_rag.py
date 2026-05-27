import asyncio
import time
import json
from app.services.legal import (
    legal_classifier,
    legal_retrieval_service,
)
from app.services.llm.router import llm_router
from app.services.legal.composition import legal_composer
from app.services.legal.validation import legal_validation_service
from app.services.legal.confidence import legal_confidence_service


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
        "question": "Um trabalhador dispensado sem justa causa em Angola tem direito a indemnização? Como se calcula a mesma e quais os prazos para contestation?",
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
        "question": "Como se processa a partilha de herança quando existem herdeiros menores em Angola? Qual a intervención do tribunal e dos pais/encarregados de educacao?",
        "complexity": "Alta",
    },
    {
        "id": 5,
        "category": "Arrendamento",
        "question": "Um senhorio pode náo renovar um contrato de arrendamento habitacional em Angola sem invocar justo motivo? O que diz a lei sobre o direito de preferencia do inquilino?",
        "complexity": "Alta",
    },
    {
        "id": 6,
        "category": "Responsabilidade Civil",
        "question": "Uma empresa que commercializa produtos defeituosos responde civilmente pelos danos causados ao consumidor em Angola? Quais os pressupostos da responsabilidade do produtor?",
        "complexity": "Alta",
    },
    {
        "id": 7,
        "category": "Processo Civil",
        "question": "O que acontece se uma das partes num processo civil em Angola nao comparece à audiencia de julgamento? O julgamento pode prosseguir à revelia?",
        "complexity": "Média",
    },
    {
        "id": 8,
        "category": "Direito Societário",
        "question": "Um socio minoritário de uma sociedade anónima em Angola pode impugnar uma deliberação da assembleia geral que considera lesiva dos seus direitos? Em que prazo e perante que órgão?",
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
        "question": "Em caso de divorcio litigioso em Angola, como se determina a guarda dos filhos menores? Que peso tem a vontade da criança na decisão do tribunal?",
        "complexity": "Alta",
    },
]


async def run_question(q: dict) -> dict:
    result = {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "complexity": q["complexity"],
        "classification": None,
        "chunks_retrieved": 0,
        "chunks_sources": [],
        "answer": None,
        "provider_used": None,
        "confidence_level": None,
        "confidence_score": None,
        "validation_issues": [],
        "latency_seconds": None,
        "error": None,
    }

    try:
        start = time.time()
        classification = await legal_classifier.classify(q["question"], [])
        result["classification"] = classification.model_dump()

        retrieval = await legal_retrieval_service.retrieve(
            q["question"],
            classification,
            conversation_history=[],
            active_document_id=None,
        )
        result["chunks_retrieved"] = len(retrieval.retrieved_chunks)
        result["chunks_sources"] = [
            {
                "title": c.title,
                "source": c.source,
                "page": c.page,
                "text_snippet": c.text[:200],
            }
            for c in retrieval.retrieved_chunks[:4]
        ]

        if not retrieval.retrieved_chunks:
            result["answer"] = "NO CONTEXT - sem chunks recuperados"
            result["latency_seconds"] = round(time.time() - start, 2)
            return result

        prompt = legal_composer.build_prompt(
            q["question"],
            classification,
            retrieval.retrieved_chunks,
            conversation_history=[],
        )
        raw_answer, provider_used = await llm_router.generate(prompt, provider=None)
        result["provider_used"] = provider_used
        answer_draft = legal_composer.parse_llm_json(raw_answer)
        answer_draft = legal_composer.constrain_draft_to_context(
            answer_draft, retrieval.retrieved_chunks
        )
        validation = legal_validation_service.validate(
            classification, retrieval, answer_draft
        )
        confidence = legal_confidence_service.score(
            classification, retrieval, validation
        )

        result["answer"] = legal_composer.compose_answer(
            classification, answer_draft, validation, confidence, []
        )
        result["confidence_level"] = confidence.level
        result["confidence_score"] = confidence.score
        result["validation_issues"] = [i.model_dump() for i in validation.issues]
        result["latency_seconds"] = round(time.time() - start, 2)

    except Exception as exc:
        result["error"] = str(exc)
        result["latency_seconds"] = round(time.time() - start, 2)

    return result


async def main():
    print("=" * 80)
    print("BENCHMARK RAG - QUESTOES JURIDICAS COMPLEXAS - ANGOLA")
    print("=" * 80)
    print()

    results = []
    for q in QUESTIONS:
        print(
            f"[{q['id']}/{len(QUESTIONS)}] {q['category']} | {q['complexity']} | {q['question'][:80]}..."
        )
        r = await run_question(q)
        results.append(r)
        status = "OK" if not r["error"] else f"ERR: {r['error'][:60]}"
        print(
            f"  -> {status} | chunks={r['chunks_retrieved']} | latency={r['latency_seconds']}s | conf={r['confidence_level']} ({r['confidence_score']})"
        )
        print()

    print("=" * 80)
    print("OVERVIEW COMPLETO")
    print("=" * 80)

    total = len(results)
    successful = sum(1 for r in results if not r["error"])
    failed = total - successful
    avg_latency = sum(r["latency_seconds"] or 0 for r in results) / total
    avg_confidence = sum(r["confidence_score"] or 0 for r in results) / total
    high_conf = sum(
        1 for r in results if r["confidence_score"] and r["confidence_score"] >= 0.7
    )
    low_conf = sum(
        1 for r in results if r["confidence_score"] and r["confidence_score"] < 0.5
    )

    print(f"\nEstatisticas Gerais:")
    print(f"  Total de questoes: {total}")
    print(f"  Com sucesso: {successful}")
    print(f"  Falhas: {failed}")
    print(f"  Latencia media: {avg_latency:.2f}s")
    print(f"  Confianca media: {avg_confidence:.2f}")
    print(f"  Alta confianca (>=0.7): {high_conf}")
    print(f"  Baixa confianca (<0.5): {low_conf}")

    print(f"\nPor Categoria:")
    for cat in set(r["category"] for r in results):
        cat_results = [r for r in results if r["category"] == cat]
        cat_avg = sum(r["confidence_score"] or 0 for r in cat_results) / len(
            cat_results
        )
        cat_chunks = sum(r["chunks_retrieved"] for r in cat_results) / len(cat_results)
        cat_latency = sum(r["latency_seconds"] or 0 for r in cat_results) / len(
            cat_results
        )
        cat_errors = sum(1 for r in cat_results if r["error"])
        print(
            f"  {cat}: conf={cat_avg:.2f} chunks={cat_chunks:.1f} latency={cat_latency:.2f}s errors={cat_errors}"
        )

    print(f"\nBenchmark Scores (0-10):")
    relevance = successful / total * 10
    accuracy = avg_confidence * 10
    speed = max(0, 10 - (avg_latency / 3))
    coverage = sum(1 for r in results if r["chunks_retrieved"] >= 3) / total * 10
    quality = (relevance + accuracy + speed + coverage) / 4
    overall = quality * 0.6 + (avg_confidence * 10) * 0.4

    print(f"  Relevancia (respostas geradas): {relevance:.1f}/10")
    print(f"  Precisao (confianca media): {accuracy:.1f}/10")
    print(f"  Velocidade (latencia): {speed:.1f}/10")
    print(f"  Cobertura (chunks >= 3): {coverage:.1f}/10")
    print(f"  Qualidade geral: {quality:.1f}/10")
    print(f"  OVERALL BENCHMARK: {overall:.1f}/10")

    print(f"\nQuestoes detalhas:")
    for r in results:
        print(f"\n[{r['id']}] {r['category']} - {r['complexity']}")
        print(f"  Q: {r['question']}")
        print(f"  Chunks recuperados: {r['chunks_retrieved']}")
        if r["chunks_sources"]:
            for i, c in enumerate(r["chunks_sources"], 1):
                print(
                    f"    [{i}] {c['title']} p.{c['page']} | {c['text_snippet'][:100]}..."
                )
        print(f"  Confianca: {r['confidence_level']} ({r['confidence_score']})")
        print(f"  Provider: {r['provider_used']}")
        print(f"  Latencia: {r['latency_seconds']}s")
        if r["validation_issues"]:
            print(f"  Validacao issues: {[i['code'] for i in r['validation_issues']]}")
        if r["error"]:
            print(f"  ERRO: {r['error']}")
        else:
            answer_preview = r["answer"][:400] if r["answer"] else ""
            print(f"  Resposta: {answer_preview}...")

    output_path = "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados detalhados guardados em: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
