# Document Intelligence Platform

Plateforme d'analyse intelligente de documents d'entreprise (rapports financiers,
contrats juridiques) basée sur une architecture RAG Multi-Agents.

## État du projet — Sprint 2

- **Sprint 1** : `ingestion/` (parsing PDF/DOCX) et `chunking/` (découpage structurel).
- **Sprint 2** : `vectorization/` — interface abstraite `VectorStore`, implémentation
  ChromaDB (embeddings ONNX locaux, sans dépendance torch).

L'orchestration multi-agents et l'API sont prévues pour les sprints suivants
(modules squelettes présents dans `src/document_intelligence/`).

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

## Docker (module ingestion uniquement)

```bash
docker build -f docker/ingestion.Dockerfile -t document-intelligence-ingestion .
```

## Hors périmètre du Sprint 2 (à venir)

- **Sprint 3** : `orchestration/` — agents LangGraph (routage, retrieval,
  synthèse, citation).
- **Sprint 4** : `api/` — FastAPI + endpoints, docker-compose multi-services,
  CI/CD GitHub Actions.
- Migration du vector store vers Qdrant/Pinecone (interface déjà abstraite,
  non implémentée).
