import asyncio
import json
from app.services.rag.pipeline import rag_pipeline
from app.services.legal.classification import legal_classifier

async def test_transformation():
    print("\n--- TESTE DE TRANSFORMAÇÃO (SIMPLIFICAÇÃO) ---")
    
    # Histórico com uma resposta técnica longa
    history = [
        "Utilizador: Trabalhei 15 anos e não me querem pagar, o que faço?",
        "IA: Se foi despedido de uma empresa onde trabalhou durante mais de 15 anos... [Art. 300, 310 da LGT]..."
    ]
    
    question = "DIGA EM TERMOS SIMPLES"
    
    print(f"\nPergunta: {question}")
    
    # 1. Testar Classificação
    classification = await legal_classifier.classify(question, history)
    print("\n[CLASSIFICAÇÃO]")
    print(f"is_transformation: {classification.is_transformation}")
    print(f"transformation_type: {classification.transformation_type}")
    print(f"search_query: {classification.search_query}")
    
    # 2. Testar Pipeline
    print("\n[PIPELINE FINAL]")
    response = await rag_pipeline.answer_query(question, conversation_history=history)
    
    print("\nRESPOSTA DA IA:")
    print("-" * 50)
    print(response.answer)
    print("-" * 50)
    print(f"Answer Mode: {response.answer_mode}")

if __name__ == "__main__":
    asyncio.run(test_transformation())
