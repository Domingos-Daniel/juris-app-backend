"""Benchmark COMPLETO: Fiabilidade, Confiabilidade, Raciocinio.
12 questoes reais - leigo a profissional. Notas 0-10."""

import json, time, httpx, asyncio, sys, io, re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "Admin123@"

QUESTIONS = [
    # ---- LEIGO (cidadao comum) ----
    (
        "Leigo",
        "Fui despedido sem justa causa. Tenho direito a indemnizacao? O que devo fazer?",
        "Leigo 1 - Despedimento",
    ),
    (
        "Leigo",
        "Estou a divorciar-me e tenho 2 filhos menores. Com quem ficam as criancas?",
        "Leigo 2 - Guarda de filhos",
    ),
    (
        "Leigo",
        "Vivo ha 20 anos num terreno dos meus pais sem documentos. Posso ser despejado?",
        "Leigo 3 - Posse de terreno",
    ),
    (
        "Leigo",
        "Perdi o meu Bilhete de Identidade. Como pec. o segunda via?",
        "Leigo 4 - Bilhete de Identidade",
    ),
    (
        "Leigo",
        "O que sao os meus direitos fundamentais na Constituicao?",
        "Leigo 5 - Direitos fundamentais",
    ),
    (
        "Leigo",
        "O meu pai faleceu e deixou uma casa. Como dividir a heranca entre os filhos?",
        "Leigo 6 - Heranca",
    ),
    # ---- PROFISSIONAL (advogado/jurista) ----
    (
        "Profissional",
        "Quais os pressupostos da responsabilidade civil extracontratual no CC Angolano e como se articula culpa com causalidade adequada?",
        "Prof 1 - Responsabilidade civil",
    ),
    (
        "Profissional",
        "Configuracao do crime de burla qualificada: elementos objetivos, subjetivos e moldura penal aplicavel.",
        "Prof 2 - Burla qualificada",
    ),
    (
        "Profissional",
        "Empresa sem pagar impostos 3 anos: ainda pode ser autuada? Prazo prescricional no CGT?",
        "Prof 3 - Prescricao tributaria",
    ),
    (
        "Profissional",
        "Pressupostos da prisao preventiva no CPP angolano e prazos maximos por fase processual.",
        "Prof 4 - Prisao preventiva",
    ),
    (
        "Profissional",
        "Como se constitui uma sociedade por quotas em Angola? Requisitos legais e capital minimo.",
        "Prof 5 - Sociedade por quotas",
    ),
    (
        "Profissional",
        "Prazo para interpor recurso contencioso de acto administrativo e fundamentos de impugnacao.",
        "Prof 6 - Recurso contencioso",
    ),
]


async def get_token():
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{BASE_URL}/auth/login",
            headers={"Content-Type": "application/json"},
            json={"username": USERNAME, "password": PASSWORD},
        )
        return r.json()["token"]


async def chat(token, question):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    payload = {
        "question": question,
        "provider": None,
        "conversation_history": [],
        "chat_id": None,
        "active_document_id": None,
    }
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=200) as c:
        r = await c.post(f"{BASE_URL}/chat", headers=headers, json=payload)
    elapsed = time.perf_counter() - start
    if r.status_code != 200:
        return {"error": r.text[:500], "latency": round(elapsed, 1)}
    d = r.json()
    ans = d.get("answer", "")
    if ans.startswith("{"):
        try:
            ans = json.loads(ans).get("rich_content", ans)
        except:
            pass
    return {
        "latency": round(elapsed, 1),
        "provider": d.get("provider_used"),
        "mode": d.get("answer_mode"),
        "branch": (d.get("classification") or {}).get("main_branch"),
        "audience": (d.get("classification") or {}).get("audience"),
        "score": (d.get("confidence") or {}).get("score", 0),
        "level": (d.get("confidence") or {}).get("level", "?"),
        "sources": len(d.get("sources", [])),
        "verified": len(d.get("verified_articles", [])),
        "legal_basis": len(d.get("legal_basis", [])),
        "issues": [
            (i.get("severity"), i.get("code"), i.get("message", "")[:100])
            for i in d.get("validation_issues", [])
        ],
        "answer": ans,
        "answer_chars": len(ans),
    }


def clean_md(text: str) -> str:
    """Strip markdown formatting for clean display."""
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


def grade(r: dict, level: str) -> dict:
    """Grade 0-10: Fiabilidade, Confiabilidade, Raciocinio."""
    ans = r.get("answer", "")
    mode = r.get("mode", "")
    score = r.get("score", 0)
    sources = r.get("sources", 0)
    verified = r.get("verified", 0)
    issues = r.get("issues", [])
    chars = r.get("answer_chars", 0)
    branch = r.get("branch", "-")
    hi_issues = sum(1 for s, _, _ in issues if s == "high")

    # Fiabilidade (0-10): a resposta fundamenta-se em leis reais?
    f = 5.0
    if mode in ("grounded", "grounded_with_caveat") and sources >= 1:
        f = 7.0
    if mode == "grounded" and sources >= 3:
        f = 8.0
    if verified >= 2 and sources >= 3:
        f = 9.0
    if verified >= 1 and sources >= 5:
        f = 9.5
    if branch in ("indeterminado", None, "-"):
        f -= 1.0
    if hi_issues >= 2:
        f -= 1.5
    if mode == "limited":
        f = max(3.0, f - 2.0)
    if not ans or chars < 200:
        f = 2.0
    if chars > 2000:
        f = min(10.0, f + 0.5)
    f = max(0, min(10, round(f, 1)))

    # Confiabilidade (0-10): o utilizador pode confiar na resposta?
    c = 5.0
    if mode == "grounded":
        c = 7.5
    elif mode == "grounded_with_caveat":
        c = 6.5
    if verified >= 1:
        c = min(10.0, c + 1.5)
    if hi_issues == 0:
        c = min(10.0, c + 0.5)
    if level == "leigo" and chars > 500:
        c = min(10.0, c + 0.3)
    if sources >= 3:
        c = min(10.0, c + 0.5)
    if mode == "limited" and sources == 0:
        c = 4.0
    if not ans or chars < 150:
        c = 2.0
    c = max(0, min(10, round(c, 1)))

    # Raciocinio (0-10): qualidade da analise juridica
    r_val = 5.0
    if chars > 3000:
        r_val = 8.0
    elif chars > 2000:
        r_val = 7.5
    elif chars > 1000:
        r_val = 6.5
    elif chars > 500:
        r_val = 5.5
    else:
        r_val = 4.0
    # Check for structured analysis markers
    has_sections = bool(
        re.search(
            r"(?i)(?:###|pressupost|elemento|requisito|prazo|artigo\s+\d|fundamento)",
            ans,
        )
    )
    has_practical = bool(
        re.search(r"(?i)(?:passo|deve|pode|dirija|conservator|tribunal|accao)", ans)
    )
    if has_sections:
        r_val = min(10.0, r_val + 1.0)
    if has_practical:
        r_val = min(10.0, r_val + 0.5)
    if mode == "grounded" and has_sections:
        r_val = min(10.0, r_val + 0.5)
    if level == "profissional" and chars > 2000 and has_sections:
        r_val = min(10.0, r_val + 0.5)
    if not ans or chars < 100:
        r_val = 1.0
    r_val = max(0, min(10, round(r_val, 1)))

    return {
        "fiabilidade": f,
        "confiabilidade": c,
        "raciocinio": r_val,
        "media": round((f + c + r_val) / 3, 1),
    }


async def main():
    print("=" * 90)
    print("BENCHMARK FINAL - FIABILIDADE, CONFIABILIDADE, RACIOCINIO")
    print("Provider: OpenRouter | Modelo: openai/gpt-4o-mini | 12 questoes reais")
    print("=" * 90)
    print()

    token = await get_token()

    results = []
    for level, question, label in QUESTIONS:
        sys.stdout.write(f"[{len(results) + 1:>2}/12] {label:<32} ... ")
        sys.stdout.flush()
        r = await chat(token, question)
        r["label"] = label
        r["level"] = level
        results.append(r)
        status = "OK" if "error" not in r else "ERR"
        print(
            f"{r.get('latency', 0):>5.1f}s | {status} | {r.get('mode', '?'):<22} | {r.get('branch', '?'):<14} | score={r.get('score', 0):.2f}"
        )

    print("\n" + "=" * 90)
    print("RESULTADOS DETALHADOS - PERGUNTA, RESPOSTA, NOTAS")
    print("=" * 90)

    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    for i, r in enumerate(ok):
        g = grade(r, r.get("level", "leigo"))
        label = r["label"]
        ans = clean_md(r.get("answer", ""))
        q_text = [q for l, q, lb in QUESTIONS if lb == label][0] if QUESTIONS else ""

        print(f"\n{'=' * 90}")
        print(f"QUESTAO {i + 1}: {label}")
        print(f"{'=' * 90}")
        print(
            f"Nivel: {r['level']} | Ramo: {r.get('branch', '?')} | Audiencia: {r.get('audience', '?')}"
        )
        print(
            f"Latencia: {r['latency']}s | Modo: {r['mode']} | Score sistema: {r['score']}"
        )
        print(
            f"Fontes: {r['sources']} | Artigos verificados: {r['verified']}/{r['legal_basis']}"
        )
        issues = r.get("issues", [])
        if issues:
            print(f"Validacao ({len(issues)}):")
            for sev, code, msg in issues:
                print(f"  [{sev}] {code}: {msg}")
        print()
        print(f"PERGUNTA:")
        print(f"  {q_text}")
        print()
        print(f"RESPOSTA DA IA ({r['answer_chars']} caracteres):")
        print(f"  {ans}")
        print()
        print(f"NOTAS (0-10):")
        print(
            f"  Fiabilidade:     {'★' * int(g['fiabilidade'])}{'☆' * (10 - int(g['fiabilidade']))}  {g['fiabilidade']}/10"
        )
        print(
            f"  Confiabilidade:  {'★' * int(g['confiabilidade'])}{'☆' * (10 - int(g['confiabilidade']))}  {g['confiabilidade']}/10"
        )
        print(
            f"  Raciocinio:      {'★' * int(g['raciocinio'])}{'☆' * (10 - int(g['raciocinio']))}  {g['raciocinio']}/10"
        )
        print(
            f"  MEDIA GERAL:     {'★' * int(g['media'])}{'☆' * (10 - int(g['media']))}  {g['media']}/10"
        )

    for r in errors:
        print(f"\n{'=' * 90}")
        print(f"ERRO: {r['label']}")
        print(f"  {r['error'][:300]}")

    print(f"\n{'=' * 90}")
    print("TABELA RESUMO - 12 QUESTOES")
    print(f"{'=' * 90}")
    print(
        f"{'Questao':<24} {'Nivel':<8} {'Lat':>5} {'Modo':<22} {'Ramo':<12} {'Sc':>5} {'Fiab':>5} {'Conf':>5} {'Rac':>5} {'Med':>5}"
    )
    print("-" * 90)
    total_f = total_c = total_r = total_m = 0.0
    for r in ok:
        g = grade(r, r.get("level", "leigo"))
        total_f += g["fiabilidade"]
        total_c += g["confiabilidade"]
        total_r += g["raciocinio"]
        total_m += g["media"]
        m = r.get("mode", "?")[:20]
        print(
            f"{r['label']:<24} {r['level']:<8} {r['latency']:>4.0f}s  {m:<22} {r.get('branch', '?'):<12} {r.get('score', 0):>4.2f} {g['fiabilidade']:>4.1f} {g['confiabilidade']:>4.1f} {g['raciocinio']:>4.1f} {g['media']:>4.1f}"
        )
    for r in errors:
        print(
            f"{r['label']:<24} {'ERRO':<8} {'--':>5}  {'---':<22} {'---':<12} {'--':>5} {'--':>5} {'--':>5} {'--':>5} {'--':>5}"
        )
    print("-" * 90)
    n_ok = len(ok)
    print(
        f"{'MEDIAS':<24} {'':<8} {'':>5}  {'':<22} {'':<12} {'':>5} {total_f / n_ok:>4.1f} {total_c / n_ok:>4.1f} {total_r / n_ok:>4.1f} {total_m / n_ok:>4.1f}"
    )

    # Save
    save_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provider": ok[0].get("provider") if ok else "?",
        "model": "openai/gpt-4o-mini via OpenRouter",
        "averages": {
            "fiabilidade": round(total_f / n_ok, 1) if n_ok else 0,
            "confiabilidade": round(total_c / n_ok, 1) if n_ok else 0,
            "raciocinio": round(total_r / n_ok, 1) if n_ok else 0,
            "media_geral": round(total_m / n_ok, 1) if n_ok else 0,
        },
        "results": [{k: v for k, v in r.items() if k != "answer"} for r in results],
    }
    with open("scratch/benchmark_completo.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSalvo em scratch/benchmark_completo.json")


if __name__ == "__main__":
    asyncio.run(main())
