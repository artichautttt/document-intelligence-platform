# Document Intelligence Platform

Plateforme d'analyse intelligente de documents d'entreprise (rapports financiers,
contrats juridiques) basée sur une architecture RAG Multi-Agents.

## État du projet — Sprint 1

Ce sprint couvre uniquement le pipeline d'**ingestion** et de **chunking**.
La vectorisation, l'orchestration multi-agents et l'API sont prévues pour les
sprints suivants (modules squelettes présents dans `src/document_intelligence/`).

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

document = ingest("rapport.pdf")
chunks = chunk_document(document)

for chunk in chunks:
    print(chunk.chunk_id, chunk.provenance.element_ids, chunk.text[:80])
```

## Docker (module ingestion uniquement)

```bash
docker build -f docker/ingestion.Dockerfile -t document-intelligence-ingestion .
```

## Hors périmètre du Sprint 1 (à venir)

- **Sprint 2** : `vectorization/` — interface abstraite de vector store,
  implémentation ChromaDB, migration future vers Qdrant/Pinecone.
- **Sprint 3** : `orchestration/` — agents LangGraph (routage, retrieval,
  synthèse, citation).
- **Sprint 4** : `api/` — FastAPI + endpoints, docker-compose multi-services,
  CI/CD GitHub Actions.
