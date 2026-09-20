# Document Intelligence Platform

[![CI](https://github.com/artichautttt/document-intelligence-platform/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/artichautttt/document-intelligence-platform/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Plateforme d'analyse intelligente de documents d'entreprise (rapports financiers,
contrats juridiques) basée sur une architecture RAG Multi-Agents.

## État du projet — Sprint 6

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

- **Sprint 6** — qualité et exploitation d'un pipeline LLM :
  - `evaluation/` — évaluation RAG (retrieval + génération), voir *Évaluation*.
  - `observability/` — tracing par étape, latence, tokens et coût, endpoint
    `GET /metrics`, voir *Observabilité*.
  - Synthèse en **sortie structurée** (function calling) avec prompt système
    few-shot ; les citations sont désormais celles que le modèle déclare, et
    non plus tous les chunks récupérés, voir *Prompting et sorties structurées*.
  - `vectorization/hybrid_store.py` — retrieval hybride (dense + BM25 par
    fusion RRF), activé par défaut (`RETRIEVAL_MODE=hybrid`).
  - Export OpenTelemetry (compatible Langfuse) et métriques persistantes en
    SQLite, voir *Observabilité*.
  - Corpus d'évaluation étendu (63 questions), voir *Évaluation*.

## Installation

```bash
uv sync
```

## Qualité

```bash
uv run ruff check .        # lint
uv run ruff format --check .  # formatage
uv run mypy               # typage strict
uv run pytest            # tests (107 tests, 1 skip)
```

Ces quatre commandes sont exécutées en CI sur chaque push/PR (voir *CI/CD*).

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
answer = answer_query(
    "Quels sont les resultats financiers du trimestre ?", store=store, llm=llm
)
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

## Prompting et sorties structurées

`orchestration/prompts.py` et `orchestration/nodes.py` :

- **Function calling** : `AnthropicLLMClient.complete_structured` force un appel
  d'outil (`tool_choice`) dont le `input_schema` est le schéma Pydantic
  `GroundedAnswer` (`answer`, `cited_chunk_ids`, `sufficient_context`). Les
  arguments sont validés par Pydantic ; tout autre `LLMClient` hérite d'un repli
  par prompt JSON (`LLMClient.complete_structured`).
- **Prompt système few-shot** : règles de fondement strict + deux exemples
  (contexte suffisant / insuffisant).
- **Citations vérifiables** : seuls les identifiants déclarés par le modèle
  *et* présents dans le contexte récupéré sont conservés ; les identifiants
  inventés sont écartés et journalisés (`synthesis.unknown_citations`).
- **Abstention** : `sufficient_context=false` (exposé dans la réponse de
  `/query`) plutôt qu'une réponse forcée sans support.
- **Injection indirecte** : extraits balisés en XML, échappés, et déclarés comme
  données dans le prompt système. Cela réduit le risque, sans l'éliminer.

## Évaluation

```bash
# Retrieval seul (sans LLM ni réseau) sur les documents de test
uv run python -m document_intelligence.evaluation --retrieval-only --k 5

# Bout en bout, avec ANTHROPIC_API_KEY (juge = même modèle par défaut)
uv run python -m document_intelligence.evaluation --k 5 --output rapport.json
```

Le jeu de test compte **63 questions** (55 répondables, 8 sans réponse dans les
documents) réparties en deux fichiers de `evaluation/golden/` :

- `fixtures_qa.jsonl` — 15 questions sur les trois documents de `tests/fixtures/` ;
- `extended_qa.jsonl` — 48 questions sur un corpus **synthétique** de trois
  documents plus longs (contrat de licence, rapport financier semestriel, audit
  de sécurité ; 10 à 13 sections chacun), généré de façon déterministe par
  `evaluation/corpus.py`. Sociétés, montants et dates sont fictifs.

La pertinence d'un chunk est définie par une sous-chaîne attendue (les
identifiants de chunks changent à chaque ingestion). Un test vérifie que chaque
passage attendu existe bien dans le corpus.

| Famille | Métriques |
|---|---|
| Retrieval | hit rate@k, recall, precision, MRR, nDCG@k |
| Génération (LLM-juge, à la RAGAS) | *faithfulness* (affirmations soutenues / total), *answer relevance* (1-5 normalisé), *citation precision* |
| Robustesse | exactitude de l'abstention sur les questions sans réponse, latence |

### Retrieval mesuré (ChromaDB, embeddings ONNX locaux, 6 documents)

| k | dense : hit rate / MRR / nDCG | hybride : hit rate / MRR / nDCG |
|---|---|---|
| 1 | 0.56 / 0.56 / 0.56 | 0.60 / 0.60 / 0.60 |
| 3 | 0.67 / 0.62 / 0.63 | 0.78 / 0.68 / 0.71 |
| 5 | 0.76 / 0.64 / 0.67 | **0.84 / 0.70 / 0.73** |

Le mode hybride (`vectorization/hybrid_store.py`) demande `4 × k` candidats au
store dense, les note avec BM25 (racines grossières, mots vides français), puis
fusionne les deux classements par *Reciprocal Rank Fusion*. Il améliore les
trois métriques à chaque valeur de k et il est activé par défaut
(`RETRIEVAL_MODE=hybrid`, `dense` pour le désactiver).

Limites à connaître :

- Le retrieval reste imparfait : à k=5, 9 questions répondables sur 55 n'ont
  aucun chunk pertinent (`hit rate` 0.84). Le modèle d'embedding local est
  petit ; un meilleur embedding ou un re-ranker par cross-encoder serait la
  suite logique.
- Le réordonnancement lexical a été conçu après avoir observé un échec du
  premier jeu (15 questions) ; le jeu élargi le confirme, mais les deux jeux
  ne sont pas indépendants du développement. Une question ambiguë a en outre été
  précisée quand le corpus s'est élargi.
- Corpus synthétique et peu volumineux : ces chiffres servent à détecter une
  régression, ce n'est pas un benchmark.
- Les métriques de génération n'ont pas encore été mesurées sur un vrai modèle
  (aucune clé API dans l'environnement de développement) ; les tests les
  valident avec un juge déterministe.

Un garde-fou de régression (`hit_rate >= 0.78`, `MRR >= 0.64` à k=5, mode
hybride) et un test « le hybride n'est pas pire que le dense » s'exécutent dans
la suite de tests, donc en CI.

## Observabilité

- **Tracing** : chaque nœud (`node.route|retrieve|synthesize`), le pipeline,
  `api.query` et chaque appel LLM (`llm.call`) sont des spans
  (`observability/tracing.py`) : durée, erreur, attributs (hits, score max,
  citations, tokens, modèle), émis en logs JSON. Un sink défaillant est
  journalisé mais ne perturbe jamais le pipeline.
- **Métriques** : `GET /metrics` (protégé par `X-API-Key` si défini) renvoie
  latence moyenne/p50/p95 par étape, et pour le LLM : appels, tokens d'entrée
  et de sortie, latence, coût estimé. Le coût vaut 0 tant que
  `LLM_INPUT_PRICE_PER_MTOK` / `LLM_OUTPUT_PRICE_PER_MTOK` ne sont pas
  renseignés (`pricing_configured` l'indique).
- **Persistance** : avec `METRICS_DB_PATH=/chemin/metrics.db`, les événements
  sont journalisés dans SQLite et l'état est restauré au redémarrage
  (rétention de 30 jours). Sans cette variable, les métriques restent en mémoire
  et sont perdues à l'arrêt. Sous Docker, monter ce fichier sur un volume.
- **Export OpenTelemetry / Langfuse** : `uv sync --group otel`, puis définir
  `OTEL_EXPORTER_OTLP_ENDPOINT` (et `OTEL_EXPORTER_OTLP_HEADERS` au format
  `k=v,k2=v2`). La hiérarchie pipeline → nœuds → appel LLM est conservée, et
  les appels LLM portent les attributs `gen_ai.*` (modèle, tokens). Pour
  Langfuse, l'endpoint OTLP est `<host>/api/public/otel` avec un en-tête
  `Authorization=Basic <base64(clé_publique:clé_secrète)>` (d'après sa
  documentation). Testé avec un exporteur OpenTelemetry en mémoire (hiérarchie,
  attributs, erreurs) ; **pas testé contre une instance Langfuse réelle**.
- **LangSmith** : le pipeline étant un `StateGraph` LangGraph, LangSmith peut
  tracer les exécutions via `LANGSMITH_TRACING` / `LANGSMITH_API_KEY`. Non
  vérifié ici.

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

`.github/workflows/ci.yml` s'exécute à chaque push/PR sur `main`, en trois jobs :

- **lint** — `ruff check`, `ruff format --check`, puis `mypy --strict` (0 erreur).
- **test** — `pytest` avec couverture, en matrice Python 3.11 et 3.12.
- **docker-build** — build des images `ingestion` et `api` via Buildx, avec cache
  de layers GitHub Actions ; ne démarre que si `lint` et `test` passent.

## Hors périmètre (à venir)

- Fine-tuning : non pertinent à ce stade. L'ancrage sur les documents passe par
  le RAG (les connaissances changent d'un dossier à l'autre), et le fine-tuning
  n'apporterait rien sans un jeu de données d'entraînement réel, que le projet
  n'a pas. La démarche serait : mesurer d'abord avec `evaluation/`, puis affiner
  le retrieval (chunking, re-ranking) avant d'envisager un modèle spécialisé.

- Migration vers un autre fournisseur LLM (OpenAI, modèle local) — l'interface
  `LLMClient` le permet déjà sans changer les appelants.
- Migration vers Pinecone (Qdrant est désormais disponible ; Pinecone resterait
  une implémentation `VectorStore` supplémentaire si besoin).
- Authentification par clé d'API unique et partagée : pas de gestion
  multi-utilisateurs, de rotation de clé ou de scopes ; à envisager si l'API
  est exposée à plusieurs clients distincts.

## Licence

MIT — voir [LICENSE](LICENSE).
