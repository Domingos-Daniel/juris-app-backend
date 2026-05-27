import sys
import os

# Ensure the app module can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.legal.classification import legal_classifier

history = [
    "Utilizador: Quais sao os direitos do trabalhador em caso de despedimento?",
    "Assistente: Em caso de despedimento, o trabalhador tem direitos...",
    "Utilizador: quais os artigos que suportam a tua respota?",
    "Assistente: Os artigos que suportam a resposta sobre os direitos do trabalhador em caso de despedimento são o artigo 248.º2 e o artigo 310.º4 da Lei Geral do Trabalho (Lei 12/23)."
]

query = "e o que devo fazer?"

print("================================================================================")
print("DIAGNÓSTICO DO FOLLOW-UP")
print("================================================================================")

import asyncio

async def run_diag():
    result = await legal_classifier.classify(query, history)

    print(f"\nFULL CLASSIFICATION:")
    print(result.model_dump_json(indent=2))

    print(f"\nSpecificity: {result.specificity}")
    print(f"Main Branch: {result.main_branch}")
    print(f"Topic Route: {result.topic_route}")
    print("================================================================================")

asyncio.run(run_diag())
