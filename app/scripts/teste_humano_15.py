"""15 perguntas simulando angolanos reais — do leigo ao complexo."""

import asyncio
import json
import time
from pathlib import Path

from app.core.auth import validate_login
from app.services.rag.pipeline import rag_pipeline

QUESTIONS = [
    # ── LEIGOS (cidadão comum) ──
    {
        "id": 1,
        "perfil": "Cidadão — Baixa escolaridade",
        "area": "Família",
        "pergunta": "O meu marido abandonou-me com 4 filhos. Ele tem obrigação de dar dinheiro para comer? Como faço para obrigar?",
        "complexidade": "Baixa",
    },
    {
        "id": 2,
        "perfil": "Trabalhador informal",
        "area": "Trabalho",
        "pergunta": "Trabalho numa cantina há 5 anos sem contrato. O patrão mandou-me embora ontem sem me pagar nada. O que posso fazer? Tenho direito a quê?",
        "complexidade": "Baixa",
    },
    {
        "id": 3,
        "perfil": "Jovem adulto",
        "area": "Identificação / Administrativo",
        "pergunta": "Perdi o meu bilhete de identidade. Como tirar a segunda via? Quanto custa e quanto tempo demora?",
        "complexidade": "Baixa",
    },
    {
        "id": 4,
        "perfil": "Estudante universitário",
        "area": "Constitucional",
        "pergunta": "A polícia pode entrar na minha casa sem autorização? O que diz a Constituição sobre a inviolabilidade do domicílio?",
        "complexidade": "Média",
    },
    {
        "id": 5,
        "perfil": "Pequeno comerciante",
        "area": "Fiscal",
        "pergunta": "Abri uma loja de roupa há 2 anos mas nunca paguei imposto. Agora quero legalizar. Vou ser multado? Quanto tempo tenho para pagar antes de prescrever?",
        "complexidade": "Média",
    },
    # ── INTERMÉDIOS ──
    {
        "id": 6,
        "perfil": "Funcionário público",
        "area": "Trabalho / Administrativo",
        "pergunta": "Sou funcionário público há 12 anos. Posso ser despedido sem processo disciplinar? Que garantias tenho?",
        "complexidade": "Média",
    },
    {
        "id": 7,
        "perfil": "Proprietário de imóvel",
        "area": "Arrendamento / Civil",
        "pergunta": "Aluguei a minha casa e o inquilino não paga há 6 meses. Como posso despejá-lo? Qual o prazo legal?",
        "complexidade": "Média",
    },
    {
        "id": 8,
        "perfil": "Vítima de crime",
        "area": "Penal",
        "pergunta": "Fui assaltado na rua e levaram-me o telefone e dinheiro. Quero apresentar queixa. O que acontece depois? O ladrão pode ser preso?",
        "complexidade": "Média",
    },
    {
        "id": 9,
        "perfil": "Empreendedor",
        "area": "Comercial / Societário",
        "pergunta": "Quero abrir uma empresa com mais dois sócios. Que tipo de sociedade devemos escolher? Quanto custa e quanto tempo demora para registar?",
        "complexidade": "Média",
    },
    {
        "id": 10,
        "perfil": "Herdeiro",
        "area": "Sucessões",
        "pergunta": "O meu pai morreu e deixou uma casa e um carro. Somos 5 irmãos. Como dividimos? Um dos meus irmãos quer vender e os outros não. O que fazer?",
        "complexidade": "Média",
    },
    # ── COMPLEXOS ──
    {
        "id": 11,
        "perfil": "Advogado júnior",
        "area": "Processo Penal",
        "pergunta": "Um arguido em prisão preventiva há 120 dias por crime de homicídio qualificado. Já expirou o prazo? Quais os fundamentos para requerer a liberdade?",
        "complexidade": "Alta",
    },
    {
        "id": 12,
        "perfil": "Jurista de empresa",
        "area": "Contratos / Civil",
        "pergunta": "Uma empresa angolana assinou um contrato de fornecimento com uma empresa portuguesa. Houve incumprimento. Qual a lei aplicável? O foro competente é o angolano?",
        "complexidade": "Alta",
    },
    {
        "id": 13,
        "perfil": "Consultor fiscal",
        "area": "Tributário / Fiscal",
        "pergunta": "Uma sociedade anónima angolana quer deduzir prejuízos fiscais de exercícios anteriores. Quais os limites legais para o reporte de prejuízos? Qual o prazo máximo?",
        "complexidade": "Alta",
    },
    {
        "id": 14,
        "perfil": "Magistrado",
        "area": "Processo Contencioso Administrativo",
        "pergunta": "Quais os prazos de impugnação de um acto administrativo lesivo de direitos subjectivos? Quando se conta o início do prazo? Há suspensão durante as férias judiciais?",
        "complexidade": "Alta",
    },
    {
        "id": 15,
        "perfil": "Activista / Direitos Humanos",
        "area": "Constitucional / Penal",
        "pergunta": "Qual é o regime jurídico da manifestação pública em Angola? É preciso autorização ou basta comunicação prévia? O que acontece se a polícia dispersar uma manifestação pacífica?",
        "complexidade": "Alta",
    },
]


async def testar_questao(q: dict, user_id: str) -> dict:
    inicio = time.time()
    try:
        resp = await asyncio.wait_for(
            rag_pipeline.answer_query(
                q["pergunta"],
                provider="deepseek",
                conversation_history=[],
                chat_id=None,
                active_document_id=None,
                user_id=user_id,
            ),
            timeout=120,
        )
        latencia = round(time.time() - inicio, 2)
        return {
            "id": q["id"],
            "perfil": q["perfil"],
            "area": q["area"],
            "complexidade": q["complexidade"],
            "pergunta": q["pergunta"],
            "modo": resp.answer_mode,
            "confianca_nivel": resp.confidence.get("level")
            if isinstance(resp.confidence, dict)
            else None,
            "confianca_score": resp.confidence.get("score")
            if isinstance(resp.confidence, dict)
            else None,
            "latencia_s": latencia,
            "num_fontes": len(resp.sources),
            "fontes": [
                {
                    "titulo": s.title,
                    "fonte": s.source,
                    "pagina": s.page,
                    "artigo": s.article_number,
                }
                for s in resp.sources[:5]
            ],
            "bases_legais": [
                {"diploma": b.get("diploma"), "artigo": b.get("article")}
                for b in (resp.legal_basis or [])[:5]
            ],
            "validacao": [i.get("code") for i in (resp.validation_issues or [])],
            "resposta": resp.answer[:600] if resp.answer else None,
            "provider": resp.provider_used,
            "erro": None,
        }
    except asyncio.TimeoutError:
        return {
            "id": q["id"],
            "perfil": q["perfil"],
            "area": q["area"],
            "complexidade": q["complexidade"],
            "pergunta": q["pergunta"],
            "erro": "TIMEOUT 120s",
            "latencia_s": round(time.time() - inicio, 2),
        }
    except Exception as e:
        return {
            "id": q["id"],
            "perfil": q["perfil"],
            "area": q["area"],
            "complexidade": q["complexidade"],
            "pergunta": q["pergunta"],
            "erro": str(e)[:300],
            "latencia_s": round(time.time() - inicio, 2),
        }


def avaliar(resultados: list[dict]) -> dict:
    total = len(resultados)
    ok = [r for r in resultados if not r.get("erro")]
    erros = [r for r in resultados if r.get("erro")]
    sucesso = len(ok)
    falhas = len(erros)

    scores = [r.get("confianca_score") or 0 for r in ok if r.get("confianca_score")]
    avg_conf = sum(scores) / max(len(scores), 1)

    latencias = [r["latencia_s"] for r in resultados if r.get("latencia_s")]
    avg_lat = sum(latencias) / max(len(latencias), 1)

    fontes_vals = [r["num_fontes"] for r in ok if r.get("num_fontes")]
    avg_fontes = sum(fontes_vals) / max(len(fontes_vals), 1)

    com_bases = sum(1 for r in ok if r.get("bases_legais"))
    sem_validacao = sum(1 for r in ok if not r.get("validacao"))

    # Nota geral (0-20)
    nota = 0.0
    nota += (sucesso / total) * 5  # taxa de sucesso (0-5)
    nota += min(avg_lat / 3, 1) * 3 * -1  # penalidade latência (0-3)
    nota += max(3 - min(avg_lat / 3, 3), 0) * 3  # bónus rapidez

    # Corrigido: fórmula composta
    pontuacao_lat = max(0, 5 - (avg_lat / 30) * 5)  # 0s=5, 30s=0
    pontuacao_conf = avg_conf * 5  # 1.0=5, 0.0=0
    pontuacao_fontes = min(avg_fontes / 5, 1) * 3  # 5 fontes=3pts
    pontuacao_bases = (com_bases / max(total, 1)) * 2  # bases legais
    pontuacao_validacao = (sem_validacao / max(total, 1)) * 2  # sem issues
    pontuacao_sucesso = (sucesso / total) * 3

    nota = (
        pontuacao_lat
        + pontuacao_conf
        + pontuacao_fontes
        + pontuacao_bases
        + pontuacao_validacao
        + pontuacao_sucesso
    )

    if nota >= 16:
        classificacao = "Excelente — pronto para producao"
    elif nota >= 12:
        classificacao = "Bom — funcional com margem de melhoria"
    elif nota >= 8:
        classificacao = "Razoavel — precisa de ajustes no retrieval ou LLM"
    elif nota >= 4:
        classificacao = "Fraco — problemas estruturais no pipeline"
    else:
        classificacao = "Insuficiente — nao utilizavel"

    return {
        "total": total,
        "sucesso": sucesso,
        "falhas": falhas,
        "taxa_sucesso_pct": round(sucesso / total * 100, 0),
        "latencia_media_s": round(avg_lat, 1),
        "confianca_media": round(avg_conf, 2),
        "fontes_medias": round(avg_fontes, 1),
        "com_bases_legais": com_bases,
        "sem_validacao_issues": sem_validacao,
        "nota_final_0_20": round(nota, 1),
        "classificacao": classificacao,
        "detalhe_pontuacao": {
            "latencia": round(pontuacao_lat, 1),
            "confianca": round(pontuacao_conf, 1),
            "fontes": round(pontuacao_fontes, 1),
            "bases_legais": round(pontuacao_bases, 1),
            "validacao_limpa": round(pontuacao_validacao, 1),
            "taxa_sucesso": round(pontuacao_sucesso, 1),
        },
    }


async def main():
    print("=" * 95)
    print("  TESTE HUMANO REAL — 15 PERGUNTAS DE ANGOLANOS (LEIGO -> COMPLEXO)")
    print("=" * 95)

    # Pre-aquecer embedding model + classificador
    from app.services.rag.embeddings import embedding_service
    from app.services.legal import legal_classifier

    embedding_service.initialize()
    print("  Modelo de embeddings carregado. A classificar teste...")
    _ = await legal_classifier.classify("teste", [])
    print("  Classificador pronto.\n")

    user = validate_login("admin", "Admin123@")
    resultados = []
    inicio_total = time.time()

    for q in QUESTIONS:
        tag = f"{'🟢' if q['complexidade'] == 'Baixa' else '🟡' if q['complexidade'] == 'Média' else '🔴'}"
        print(f"\n[{q['id']:2d}/15] {tag} {q['perfil'][:45]:45s} | {q['area']:20s}")
        print(f"       {q['pergunta'][:95]}...")
        r = await testar_questao(q, user["id"])
        resultados.append(r)
        if r.get("erro"):
            print(f"       ❌ ERRO: {r['erro'][:100]}")
        else:
            bases = ", ".join(
                f"{b['artigo'] or '?'}" for b in (r.get("bases_legais") or [])[:3]
            )
            print(
                f"       ✅ {r['modo']} | conf={r['confianca_score']:.2f} | {r['latencia_s']}s | {r['num_fontes']} fontes | bases=[{bases}]"
            )

    tempo_total = round(time.time() - inicio_total, 1)
    aval = avaliar(resultados)

    print("\n" + "=" * 95)
    print("  AVALIACAO FINAL")
    print("=" * 95)
    print(
        f"  Total: {aval['total']} | Sucesso: {aval['sucesso']} | Falhas: {aval['falhas']} ({aval['taxa_sucesso_pct']:.0f}%)"
    )
    print(
        f"  Latencia media: {aval['latencia_media_s']}s | Tempo total: {tempo_total}s"
    )
    print(
        f"  Confianca media: {aval['confianca_media']} | Fontes medias: {aval['fontes_medias']}"
    )
    print(
        f"  Com bases legais: {aval['com_bases_legais']}/{aval['total']} | Sem issues validacao: {aval['sem_validacao_issues']}/{aval['total']}"
    )
    print(f"\n  ╔══════════════════════════════════╗")
    print(
        f"  ║  NOTA: {aval['nota_final_0_20']:.1f} / 20  —  {aval['classificacao'][:33]:33s} ║"
    )
    print(f"  ╚══════════════════════════════════╝")
    print(f"\n  Distribuicao:")
    for k, v in aval["detalhe_pontuacao"].items():
        bar = "█" * int(v * 2)
        print(f"    {k:20s} {v:4.1f}/5  {bar}")

    # Por complexidade
    for nivel in ["Baixa", "Média", "Alta"]:
        do_nivel = [r for r in resultados if r["complexidade"] == nivel]
        if not do_nivel:
            continue
        ok_nivel = [r for r in do_nivel if not r.get("erro")]
        confs = [r.get("confianca_score") or 0 for r in ok_nivel]
        print(
            f"\n  {nivel:8s}: {len(ok_nivel)}/{len(do_nivel)} ok | conf media={sum(confs) / max(len(confs), 1):.2f}"
        )

    # Mostrar 3 melhores e 3 piores
    ok_sorted = sorted(
        [r for r in resultados if not r.get("erro")],
        key=lambda r: r.get("confianca_score") or 0,
        reverse=True,
    )
    print(f"\n  TOP 3 respostas:")
    for r in ok_sorted[:3]:
        print(
            f"    [{r['id']}] {r['area']:20s} conf={r['confianca_score']:.2f} | {r['pergunta'][:70]}..."
        )
    print(f"\n  PIORES 3 respostas:")
    for r in ok_sorted[-3:]:
        print(
            f"    [{r['id']}] {r['area']:20s} conf={r['confianca_score']:.2f} | {r['pergunta'][:70]}..."
        )

    # Guardar
    out = Path("C:/Projectos/TCC/backend/data/processed/teste_humano_15.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"avaliacao": aval, "resultados": resultados}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\n  Resultados guardados: {out}")


if __name__ == "__main__":
    asyncio.run(main())
