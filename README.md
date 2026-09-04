# Document Intelligence Platform

Plateforme d'analyse intelligente de documents d'entreprise (rapports financiers,
contrats juridiques) basée sur une architecture RAG Multi-Agents.

## État du projet — Sprint 3

- **Sprint 1** : `ingestion/` (parsing PDF/DOCX) et `chunking/` (découpage structurel).
- **Sprint 2** : `vectorization/` — interface abstraite `VectorStore`, implémentation
  ChromaDB (embeddings ONNX locaux, sans dépendance torch).
- **Sprint 3** : `orchestration/` — pipeline d'agents routage → retrieval → synthèse
  citée, construit comme une suite de nœuds `AgentState -> AgentState`
  (`orchestration/nodes.py`) afin d'être portable vers un `StateGraph` LangGraph
  sans changer la logique métier. Interface abstraite `LLMClient` (même principe
  que `VectorStore`) ; aucun fournisseur LLM concret n'est encore branché.

L'API est prévue pour le sprint suivant (module squelette présent dans
`src/document_intelligence/`).

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

## Docker (module ingestion uniquement)

```bash
docker build -f docker/ingestion.Dockerfile -t document-intelligence-ingestion .
```

## Hors périmètre du Sprint 3 (à venir)

- **Sprint 4** : `api/` — FastAPI + endpoints, docker-compose multi-services,
  CI/CD GitHub Actions.
- Implémentation concrète de `LLMClient` (Anthropic/OpenAI) et ajout de la
  dépendance LangGraph pour porter `orchestration/nodes.py` sur un vrai
  `StateGraph`.
- Migration du vector store vers Qdrant/Pinecone (interface déjà abstraite,
  non implémentée).
