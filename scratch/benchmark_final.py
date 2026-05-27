"""Benchmark final - 6 questoes, piramyd gpt-5.4, todas correcoes."""

import json, time, httpx, asyncio, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "Admin123@"

QUESTIONS = {
    "Leigo 1 (Trabalho)": "O que devo fazer se fui despedido do meu trabalho sem justa causa? Tenho direito a alguma indemnizacao?",
    "Leigo 2 (Familia)": "Estou a divorciar-me e tenho dois filhos menores. Com quem ficam as criancas? A vontade deles conta para o tribunal?",
    "Leigo 3 (Terras)": "Vivo numa casa ha 20 anos num terreno que era dos meus pais. Nao tenho documentos. Posso ser despejado? O que devo fazer para legalizar?",
    "Prof. 1 (Civil)": "Quais sao os pressupostos da responsabilidade civil extracontratual previstos no Codigo Civil Angolano, e como se articula o regime da culpa com a causalidade adequada na fixacao da indemnizacao por danos futuros?",
    "Prof. 2 (Penal)": "Em que circunstancias se configura o crime de burla qualificada em Angola? Quais sao os elementos objetivos e subjetivos do tipo e qual a moldura penal aplicavel?",
    "Prof. 3 (Fiscal)": "Uma empresa angolana que deixou de pagar impostos durante 3 anos pode ainda ser autuada pela Autoridade Tributaria? Qual o prazo prescricional aplicavel segundo o Codigo Geral Tributario?",
}


async def get_token():
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{BASE_URL}/auth/login",
            headers={"Content-Type": "application/json"},
            json={"username": USERNAME, "password": PASSWORD},
        )
        return r.json()["token"]


async def test(token, question, label):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    payload = {
        "question": question,
        "provider": None,
        "conversation_history": [],
        "chat_id": None,
        "active_document_id": None,
    }
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"{BASE_URL}/chat", headers=headers, json=payload)
    elapsed = time.perf_counter() - start
    res = {
        "label": label,
        "question": question,
        "status_code": r.status_code,
        "latency_seconds": round(elapsed, 3),
    }
    if r.status_code != 200:
        res["error"] = r.text[:500]
        return res
    d = r.json()
    res.update(
        {
            "answer": d.get("answer", ""),
            "answer_length_chars": len(d.get("answer", "")),
            "answer_mode": d.get("answer_mode"),
            "provider_used": d.get("provider_used"),
            "num_sources": len(d.get("sources", [])),
            "confidence": d.get("confidence") or {},
            "validation_issues": d.get("validation_issues", []),
            "legal_basis_count": len(d.get("legal_basis", [])),
            "verified_articles": d.get("verified_articles", []),
            "classification_branch": (d.get("classification") or {}).get("main_branch"),
            "classification_audience": (d.get("classification") or {}).get("audience"),
        }
    )
    return res


def evaluate(t):
    conf = t.get("confidence", {})
    score = conf.get("score", 0)
    mode = t.get("answer_mode", "N/A")
    issues = t.get("validation_issues", [])
    s = t.get("num_sources", 0)
    v = len(t.get("verified_articles", []))
    al = t.get("answer_length_chars", 0)
    if score >= 0.8 and s >= 3 and v >= 1:
        q = "EXCELENTE"
    elif score >= 0.5 or (s >= 2 and al > 500):
        q = "BOA"
    elif al > 200:
        q = "MEDIA"
    else:
        q = "BAIXA"
    hi = [i for i in issues if i.get("severity") == "high"]
    if mode in ("grounded", "grounded_with_caveat") and score >= 0.6 and not hi:
        r = "ALTA"
    elif mode in ("grounded", "grounded_with_caveat") or (s >= 3 and not hi):
        r = "MEDIA"
    else:
        r = "BAIXA"
    return {
        "quality": q,
        "reliability": r,
        "score": score,
        "mode": mode,
        "sources": s,
        "verified": v,
        "hi_issues": len(hi),
    }


async def main():
    print("=" * 80)
    print(
        "BENCHMARK FINAL - 6 QUESTOES | Modelo: piramyd gpt-5.4 | Prompt: advogado senior"
    )
    print("=" * 80)
    print()

    token = await get_token()
    print(f"Token: {token[:20]}...\n")

    results = {}
    for i, (label, q) in enumerate(QUESTIONS.items(), 1):
        short = label[:40]
        sys.stdout.write(f"[{i}/6] {short}... ")
        sys.stdout.flush()
        r = await test(token, q, label)
        results[label] = r
        st = "OK" if r["status_code"] == 200 else f"ERR {r['status_code']}"
        print(f"{r['latency_seconds']}s | {st}")

    print("\n" + "=" * 80)
    print("RESULTADOS")
    print("=" * 80)

    for label, t in results.items():
        if t.get("error"):
            print(f"\n--- [{label}] ERRO ---\n  {t['error'][:300]}")
            continue
        e = evaluate(t)
        conf = t.get("confidence", {})
        print(f"\n--- [{label}] ---")
        print(
            f"  {t['latency_seconds']}s | {t['answer_mode']} | Provider: {t['provider_used']}"
        )
        print(
            f"  Ramo: {t.get('classification_branch', '?')} | Aud: {t.get('classification_audience', '?')} | Fontes: {t['num_sources']} | Resposta: {t['answer_length_chars']} chars"
        )
        print(
            f"  QUALIDADE: {e['quality']} | CONFIABILIDADE: {e['reliability']} | Score: {e['score']} ({conf.get('level', '?')})"
        )
        print(
            f"  Verificados: {e['verified']}/{t.get('legal_basis_count', 0)} | Issues alta: {e['hi_issues']}"
        )
        issues = t.get("validation_issues", [])
        if issues:
            for iss in issues:
                print(
                    f"    [{iss.get('severity', '?')}] {iss.get('code')}: {iss.get('message', '')[:120]}"
                )
        ans = t.get("answer", "")
        if ans:
            if len(ans) > 1000:
                print(f"  Resposta: {ans[:1000]}...")
            else:
                print(f"  Resposta: {ans}")

    print("\n" + "=" * 80)
    print(
        f"{'Questao':<22} {'Lat':<8} {'Modo':<18} {'Qual':<10} {'Conf':<10} {'S':<3} {'Score':<6}"
    )
    print("-" * 80)
    for label, t in results.items():
        if t.get("error"):
            print(
                f"{label:<22} {'ERRO':<8} {'-':<18} {'-':<10} {'-':<10} {'-':<3} {'-':<6}"
            )
            continue
        e = evaluate(t)
        m = t["answer_mode"][:17]
        print(
            f"{label:<22} {str(t['latency_seconds']) + 's':<8} {m:<18} {e['quality']:<10} {e['reliability']:<10} {str(e['sources']):<3} {e['score']:<6}"
        )
    print("-" * 80)

    ok = [t for t in results.values() if not t.get("error")]
    avg = sum(evaluate(t)["score"] for t in ok) / len(ok) if ok else 0
    avgl = sum(t["latency_seconds"] for t in ok) / len(ok) if ok else 0
    print(
        f"\n  MEDIA Score: {avg:.2f} | MEDIA Latencia: {avgl:.1f}s | {len(ok)}/6 OK\n"
    )

    with open("scratch/benchmark_final.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "provider": ok[0].get("provider_used", "?") if ok else "?",
                "tests": {k: v for k, v in results.items()},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("Salvo em scratch/benchmark_final.json")


if __name__ == "__main__":
    asyncio.run(main())
