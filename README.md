# Document Intelligence Platform

Plateforme d'analyse intelligente de documents d'entreprise (rapports financiers,
contrats juridiques) basée sur une architecture RAG Multi-Agents.

## État du projet — Sprint 5

- **Sprint 1** : `ingestion/` (parsing PDF/DOCX) et `chunking/` (découpage structurel).
- **Sprint 2** : `vectorization/` — interface abstraite `VectorStore`, implémentation
  ChromaDB (embeddings ONNX locaux, sans dépendance torch).
- **Sprint 3** : `orchestration/` — pipeline d'agents routage → retrieval → synthèse
  citée, construit comme une suite de nœuds `AgentState -> AgentState`
  (`orchestration/nodes.py`). Interface abstraite `LLMClient` (même principe
  que `VectorStore`).
- **Sprint 4** : `api/` — API FastAPI exposant l'ingestion (`POST /documents`) et
  la recherche/synthèse citée (`POST /query`) au-dessus des modules des sprints
  précédents ; image Docker dédiée et `docker-compose.yml` ; CI GitHub Actions
  (tests + build des images Docker).
- **Sprint 5** :
  - `orchestration/anthropic_client.py` — implémentation concrète de `LLMClient`
    sur l'API Messages d'Anthropic ; `/query` répond désormais si
    `ANTHROPIC_API_KEY` est configuré (sinon toujours `503`).
  - `orchestration/graph.py` — le pipeline est maintenant un `StateGraph`
    LangGraph dont les nœuds sont directement les fonctions de
    `orchestration/nodes.py`, sans changement de logique métier.
  - `vectorization/qdrant_store.py` — seconde implémentation de `VectorStore`
    sur Qdrant (embeddings ONNX partagés avec ChromaDB) ; sélectionnable via
    `VECTOR_STORE_BACKEND=qdrant`. Un service `qdrant` a été ajouté à
    `docker-compose.yml`.
  - Authentification par clé d'API (en-tête `X-API-Key`) sur `/documents` et
    `/query`, activée en définissant `API_KEY` ; désactivée par défaut (mode
    développement) si `API_KEY` n'est pas configuré. `/health` reste public.

## Installation

```bash
uv sync
```

## Tests

```bash
uv run pytest
```

## Utilisation

```python
from document_intelligence.ingestion.service import ingest
from document_intelligence.chunking.service import chunk_document
from document_intelligence.vectorization.chroma_store import ChromaStore

document = ingest("rapport.pdf")
chunks = chunk_document(document)

store = ChromaStore()
store.add_chunks(chunks)

for result in store.query("resultats financiers du trimestre", k=3):
    print(result.score, result.provenance.element_ids, result.text[:80])
```

Orchestration (avec le client Anthropic, ou toute autre implémentation de `LLMClient`) :

```python
from document_intelligence.orchestration.anthropic_client import AnthropicLLMClient
from document_intelligence.orchestration.graph import answer_query

llm = AnthropicLLMClient(api_key="sk-ant-...")
answer = answer_query("Quels sont les resultats financiers du trimestre ?", store=store, llm=llm)
print(answer.answer)
for citation in answer.citations:
    print(citation.chunk_id, citation.provenance.element_ids)
```

## API

```bash
uv run uvicorn document_intelligence.api.app:app --reload
```

- `GET /health` — sonde de disponibilité (publique, sans clé d'API).
- `POST /documents` — upload multipart d'un fichier PDF/DOCX ; ingestion, chunking
  et vectorisation, retourne `{document_id, chunk_count}`.
- `POST /query` — `{"query": "...", "k": 5}` ; exécute le pipeline d'orchestration
  et retourne la réponse citée. Répond `503` tant qu'aucun `LLMClient` concret
  n'est configuré (`ANTHROPIC_API_KEY` non défini, ou `create_app(llm=...)`
  explicitement à `None`).

`/documents` et `/query` exigent l'en-tête `X-API-Key: <valeur>` dès que
`API_KEY` est défini dans l'environnement ; sinon (mode développement),
aucune authentification n'est requise.

## Docker

```bash
docker build -f docker/ingestion.Dockerfile -t document-intelligence-ingestion .
docker build -f docker/api.Dockerfile -t document-intelligence-api .
docker compose up --build
```

`docker-compose.yml` définit désormais deux services : `api` et `qdrant`. Par
défaut `api` utilise toujours ChromaDB embarqué (`VECTOR_STORE_BACKEND=chroma`) ;
passer `VECTOR_STORE_BACKEND=qdrant` bascule sur le service `qdrant` sans
changer le contrat `VectorStore`.

## CI/CD

`.github/workflows/ci.yml` exécute la suite de tests puis construit les deux
images Docker (`ingestion`, `api`) à chaque push/PR sur `main`.

## Hors périmètre (à venir)

- Migration vers un autre fournisseur LLM (OpenAI, modèle local) — l'interface
  `LLMClient` le permet déjà sans changer les appelants.
- Migration vers Pinecone (Qdrant est désormais disponible ; Pinecone resterait
  une implémentation `VectorStore` supplémentaire si besoin).
- Authentification par clé d'API unique et partagée : pas de gestion
  multi-utilisateurs, de rotation de clé ou de scopes ; à envisager si l'API
  est exposée à plusieurs clients distincts.
