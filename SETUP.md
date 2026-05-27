# Setup — Backend TCC (Postgres)

## Pré-requisitos

- Python 3.11+
- Docker Desktop (ou Postgres 17 + pgvector instalado nativamente)

## 1. Subir o Postgres com pgvector

```powershell
# Na raiz do projecto:
docker compose up -d
```

Isto sobe um container `tcc-postgres` em `127.0.0.1:5432` com:
- `POSTGRES_USER=postgres`
- `POSTGRES_PASSWORD=postgres`
- `POSTGRES_DB=tcc`
- Extensão `pgvector` incluída na imagem `pgvector/pgvector:pg17`

Verificar que está saudável:
```powershell
docker compose ps
# STATUS deve ser "healthy"
```

## 2. Criar o ambiente virtual e instalar dependências

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Inicializar o schema do Postgres

```powershell
python -m app.scripts.init_postgres
# Output esperado: "Postgres schema initialized successfully."
```

Isto cria todas as tabelas (users, chats, documents, legal_documents, legal_segments, etc.), índices e o utilizador local `admin`.

## 4. Importar o catálogo lex.ao

```powershell
python -m app.scripts.import_lex_ao_catalog
# Output esperado: "Imported 2500 lex.ao catalog records into Postgres."
```

## 5. Arrancar a API

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Ou via PowerShell:
```powershell
.\start.ps1
```

A API fica disponível em:
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/api-docs`

## 6. Correr os testes

```powershell
pytest tests\ -v
```

Nota: os testes de pipeline (`test_pipeline.py`) e golden (`test_golden.py`) requerem:
- Postgres com o schema inicializado
- Catálogo lex.ao importado
- Embeddings (modelo local ou API OpenAI)

## 7. Parar o Postgres

```powershell
docker compose down
```

Para apagar os dados persistentes:
```powershell
docker compose down -v
```

## Configuração

As variáveis de ambiente estão em `backend/.env`. As críticas para Postgres:

| Variável | Valor | Descrição |
|---|---|---|
| `POSTGRES_DSN` | `postgresql://postgres:postgres@127.0.0.1:5432/tcc` | Connection string |
| `POSTGRES_SCHEMA` | `public` | Schema PostgreSQL |
| `PGVECTOR_ENABLED` | `true` | Activa a extensão pgvector |
