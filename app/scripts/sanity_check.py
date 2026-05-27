import sys
from fastapi.testclient import TestClient
from app.main import app

def run_sanity_check():
    client = TestClient(app)
    headers = {"Authorization": "Bearer 12345678901234567890123456789012"}
    
    print("Enviando pergunta 'O que e um contrato de mutuo?' (aguardando LLM)...")
    payload = {"question": "O que e um contrato de mutuo?"}
    resp = client.post("/chat", json=payload, headers=headers)
    if resp.status_code != 200:
        print(f"Erro ao obter resposta: {resp.text}")
        sys.exit(1)
        
    data = resp.json()
    
    print("\n--- Resultado do Sanity Check ---")
    print(f"Answer Mode: {data.get('answer_mode')}")
    print(f"Total Sources: {len(data.get('legal_basis', []))}")
    print(f"Preview da Resposta: {data.get('answer', '')[:100]}...")
    
    if data.get('answer_mode') in ['grounded', 'limited']:
        print("\n[OK] Sanity check passou! A pipeline RAG processou de forma deterministica.")
        sys.exit(0)
    else:
        print("\n[X] Sanity check falhou! Modo de resposta recusado ou invalido.")
        sys.exit(1)

if __name__ == "__main__":
    run_sanity_check()
