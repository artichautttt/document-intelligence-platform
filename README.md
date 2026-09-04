# Document Intelligence Platform

Plateforme d'analyse intelligente de documents d'entreprise (rapports financiers,
contrats juridiques) basée sur une architecture RAG Multi-Agents.

## État du projet — Sprint 4

- **Sprint 1** : `ingestion/` (parsing PDF/DOCX) et `chunking/` (découpage structurel).
- **Sprint 2** : `vectorization/` — interface abstraite `VectorStore`, implémentation
  ChromaDB (embeddings ONNX locaux, sans dépendance torch).
- **Sprint 3** : `orchestration/` — pipeline d'agents routage → retrieval → synthèse
  citée, construit comme une suite de nœuds `AgentState -> AgentState`
  (`orchestration/nodes.py`) afin d'être portable vers un `StateGraph` LangGraph
  sans changer la logique métier. Interface abstraite `LLMClient` (même principe
  que `VectorStore`) ; aucun fournisseur LLM concret n'est encore branché.
- **Sprint 4** : `api/` — API FastAPI exposant l'ingestion (`POST /documents`) et
  la recherche/synthèse citée (`POST /query`) au-dessus des modules des sprints
  précédents ; image Docker dédiée et `docker-compose.yml` ; CI GitHub Actions
  (tests + build des images Docker).

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

Orchestration (nécessite une implémentation concrète de `LLMClient`, non fournie) :

```python
from document_intelligence.orchestration.graph import answer_query

answer = answer_query("Quels sont les resultats financiers du trimestre ?", store=store, llm=my_llm_client)
print(answer.answer)
for citation in answer.citations:
    print(citation.chunk_id, citation.provenance.element_ids)
```

## API

```bash
uv run uvicorn document_intelligence.api.app:app --reload
```

- `GET /health` — sonde de disponibilité.
- `POST /documents` — upload multipart d'un fichier PDF/DOCX ; ingestion, chunking
  et vectorisation, retourne `{document_id, chunk_count}`.
- `POST /query` — `{"query": "...", "k": 5}` ; exécute le pipeline d'orchestration
  et retourne la réponse citée. Répond `503` tant qu'aucun `LLMClient` concret
  n'est configuré (`create_app(llm=...)`).

## Docker

```bash
docker build -f docker/ingestion.Dockerfile -t document-intelligence-ingestion .
docker build -f docker/api.Dockerfile -t document-intelligence-api .
docker compose up --build
```

`docker-compose.yml` ne définit pour l'instant que le service `api` : ChromaDB
tourne embarqué dans le processus (`PersistentClient` sur un volume partagé).
Un service de base vectorielle dédié s'ajoutera lors de la migration vers
Qdrant/Pinecone sans changer le contrat `VectorStore`.

## CI/CD

`.github/workflows/ci.yml` exécute la suite de tests puis construit les deux
images Docker (`ingestion`, `api`) à chaque push/PR sur `main`.

## Hors périmètre du Sprint 4 (à venir)

- Implémentation concrète de `LLMClient` (Anthropic/OpenAI) et ajout de la
  dépendance LangGraph pour porter `orchestration/nodes.py` sur un vrai
  `StateGraph`.
- Migration du vector store vers Qdrant/Pinecone (interface déjà abstraite,
  non implémentée) — introduira un second service dans `docker-compose.yml`.
- Authentification/autorisation sur l'API, actuellement ouverte.
