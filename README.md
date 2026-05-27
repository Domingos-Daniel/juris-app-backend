# Sistema Inteligente de Assistencia Juridica Angolana - Backend

Backend FastAPI com RAG local para perguntas juridicas baseadas na legislacao angolana, com ingestao profissional de PDFs oficiais, OCR por pagina, embeddings OpenAI e persistencia vetorial em ChromaDB.

## O que foi implementado

- API `POST /chat` com RAG obrigatorio antes da resposta.
- API `POST /docs/ingest` para baixar e ingerir a legislacao oficial configurada no codigo.
- Seleccao de provedor LLM entre `piramyd` e `openai`.
- LLM por omissao controlado por `DEFAULT_LLM_PROVIDER`.
- Embeddings com `text-embedding-3-small`.
- Vector store persistente com `ChromaDB` em disco.
- Extracao hibrida de PDF com `PyMuPDF` e fallback OCR com `pytesseract + pdf2image`.
- Chunking semantico com `RecursiveCharacterTextSplitter`, priorizando `Artigo`, `Capitulo` e `Seccao`.
- Metadados por chunk com `source`, `title`, `link_original`, `page`, `article_number`, `law_status` e `used_ocr`.
- Persistencia minima em SQLite para historico de perguntas e respostas.

## Fontes oficiais ingeridas

- Constituicao da Republica de Angola (2022)
- Codigo Penal (Lei 38/20)
- Lei Geral do Trabalho (Lei 12/23)
- Codigo Civil

Os ficheiros sao baixados automaticamente para `data/raw_pdfs` e nao sao descarregados novamente se ja existirem localmente.

## Como rodar

1. Criar e activar o ambiente virtual:

```powershell
cd C:\Projectos\TCC\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3. Configurar `backend/.env`.

Variaveis principais:

- `DEFAULT_LLM_PROVIDER=openai` ou `piramyd`
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4o-mini`
- `OPENAI_EMBEDDING_MODEL=text-embedding-3-small`
- `PIRAMYD_API_KEY=...`
- `PIRAMYD_MODEL=minimax-m2.7:free`
- `CHROMA_DB_PATH=C:/Projectos/TCC/backend/data/vetores_legislacao`
- `POSTGRES_DSN=postgresql://user:password@localhost:5432/tcc`
- `POSTGRES_SCHEMA=public`
- `PGVECTOR_ENABLED=true`

4. Garantir OCR no Windows.

- Instalar Tesseract OCR com suporte a portugues; ou usar os artefactos locais do projecto.
- Instalar Poppler; ou usar `backend/tools/poppler`.
- Se necessario, definir `TESSERACT_CMD`, `TESSERACT_DATA_DIR` e `POPPLER_BIN_PATH`.

5. Executar a ingestao profissional da legislacao:

```powershell
python -m app.scripts.ingest_legislation
```

6. Iniciar o servidor:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /health`
- `POST /docs/ingest`
- `POST /chat`

Exemplo `POST /chat`:

```json
{
  "question": "Quais sao os requisitos legais para despedimento por justa causa na Lei Geral do Trabalho?",
  "provider": "openai"
}
```

Resposta esperada:

```json
{
  "answer": "...",
  "sources": [
    {
      "title": "Lei Geral do Trabalho (Lei 12/23)",
      "source": "Lei-Geral-do-Trabalho-Lei-12-23.pdf",
      "link_original": "https://www.lgt.gov.ao/wp-content/uploads/2025/05/Livro-da-Lei-Geral-do-Trabalho.pdf",
      "page": 10,
      "article_number": "...",
      "law_status": "Indicada como vigente no nome do ficheiro",
      "excerpt": "..."
    }
  ],
  "provider_used": "openai"
}
```

## Notas tecnicas

- O sistema nao deve responder sem contexto recuperado do indice vetorial.
- A escolha do provedor pode ser feita por `DEFAULT_LLM_PROVIDER` ou por request com o campo `provider`.
- A ingestao reinicializa a coleccao Chroma antes de indexar novamente os documentos oficiais.
- A extraccao do numero do artigo e heuristica por regex, sem hardcode de artigos especificos.
- O campo `law_status` ainda e uma inferencia simples a partir do ficheiro; para producao juridica forte, convem enriquecer isso com metadados legislativos validados.
