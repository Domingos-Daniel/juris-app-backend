"""Perfil de latencia — identifica onde o pipeline gasta tempo."""

import asyncio
import time
from app.core.auth import validate_login
from app.services.legal import legal_classifier, legal_retrieval_service
from app.services.rag.embeddings import embedding_service
from app.services.rag.retriever import retriever_service
from app.services.rag.vector_store import legislation_vector_store
from app.core.logger import configure_logging, get_logger

logger = get_logger(__name__)


async def profilar(query: str, user_id: str) -> None:
    configura_logging()
    t_total = time.time()

    # Step 1: classify
    t0 = time.time()
    classification = await legal_classifier.classify(query, [])
    t_cls = time.time() - t0
    print(f"  [1] classify                  {t_cls:5.1f}s")

    # Step 2: embed
    t0 = time.time()
    emb = await embedding_service.embed_query(query)
    t_emb = time.time() - t0
    print(f"  [2] embed_query               {t_emb:5.1f}s  | dims={len(emb)}")

    # Step 3: retrieve (single, sem expansion de queries)
    t0 = time.time()
    chunks = await retriever_service.retrieve(query, k=10)
    t_ret = time.time() - t0
    print(f"  [3] retriever_service.retrieve {t_ret:5.1f}s  | chunks={len(chunks)}")

    # Step 4: legal_retrieval (full, com build_queries)
    t0 = time.time()
    retrieval = await legal_retrieval_service.retrieve(
        query, classification, conversation_history=[], active_document_id=None
    )
    t_full_retrieval = time.time() - t0
    print(
        f"  [4] legal_retrieval (full)     {t_full_retrieval:5.1f}s  | official={len(retrieval.official_evidence)} mixed={len(retrieval.retrieved_chunks)}"
    )

    t_total = time.time() - t_total
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL (sem LLM)               {t_total:5.1f}s")
    print(
        f"  Overhead restante             {t_total - t_cls - t_emb - t_ret:5.1f}s  (scoring, dedup, filtros)"
    )
    print()


def configura_logging():
    import logging

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


async def main():
    configure_logging()
    user = validate_login("admin", "Admin123@")

    # Pre-aquecer
    print("Pre-aquecendo modelos...")
    embedding_service.initialize()
    _ = await legal_classifier.classify("teste", [])
    _ = await retriever_service.retrieve("teste", k=4)
    print("Pronto.\n")

    perguntas = [
        "Um trabalhador dispensado sem justa causa tem direito a indemnizacao?",
        "Quais os crimes de burla no codigo penal angolano?",
        "O que diz a constituicao sobre liberdade de expressao?",
        "Quais os prazos de prescricao de impostos em Angola?",
    ]

    for i, q in enumerate(perguntas, 1):
        print(f"[Q{i}] {q[:70]}...")
        await profilar(q, user["id"])
        print()


if __name__ == "__main__":
    asyncio.run(main())
