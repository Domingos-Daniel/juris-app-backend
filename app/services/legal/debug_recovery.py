import asyncio
import json
from app.services.rag.pipeline import rag_pipeline
from app.services.legal.classification import legal_classifier

async def test_recovery():
    print("\n--- TESTE DE RECUPERAÇÃO DE INTENÇÃO ---")
    
    # Simulação do histórico onde a IA alucinou gravidez
    history = [
        "Utilizador: Quais sao os direitos do trabalhador em caso de despedimento?",
        "IA: [Resposta sobre despedimento]",
        "Utilizador: e o que devo fazer?",
        "IA: [Resposta alucinando que a utilizadora está grávida baseada no Art. 30]"
    ]
    
    question = "EU NÃO ESTOU GRAVIDA"
    
    print(f"\nPergunta: {question}")
    
    # 1. Testar Classificação (Router)
    classification = await legal_classifier.classify(question, history)
    print("\n[CLASSIFICAÇÃO]")
    print(f"is_follow_up: {classification.is_follow_up}")
    print(f"is_correction: {classification.is_correction}")
    print(f"main_branch: {classification.main_branch}")
    print(f"search_query: {classification.search_query}")
    
    # 2. Testar Pipeline Completo
    print("\n[PIPELINE FINAL]")
    response = await rag_pipeline.answer_query(question, conversation_history=history)
    
    print("\nRESPOSTA DA IA:")
    print("-" * 50)
    print(response.answer)
    print("-" * 50)
    print(f"Answer Mode: {response.answer_mode}")
    print(f"Confidence: {response.confidence['level']} ({response.confidence['score']})")

if __name__ == "__main__":
    asyncio.run(test_recovery())
