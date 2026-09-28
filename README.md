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
- La génération a été mesurée une fois sur les 63 questions (voir ci-dessous),
  avec un juge identique au générateur.

### Génération mesurée (`claude-sonnet-5`, 63 questions, k=5, mode hybride)

Un seul run complet. 55 questions répondables, 8 sans réponse dans les documents.

| Métrique | Valeur | Sur |
|---|---|---|
| hit rate / MRR / nDCG (retrieval) | 0.84 / 0.70 / 0.74 | 55 répondables |
| faithfulness | 1.00 | 44 réponses jugées |
| answer relevance | 0.90 | 39 réponses (voir ci-dessous) |
| citation precision | 0.97 | réponses citant au moins un chunk |
| questions sans réponse refusées | 8 / 8 | 8 |
| refus injustifié (passage dans le contexte) | 0.04 (2 / 46) | répondables bien retrouvées |
| latence moyenne (retrieval + synthèse, hors juge) | 3.6 s | 63 |

Sur les 55 questions répondables, le modèle a refusé de répondre 11 fois : 9 de
ces refus correspondent à un échec de retrieval (le passage attendu n'était pas
dans les 5 chunks, le refus est donc justifié par le contexte reçu), et 2 sont
des refus injustifiés. **La qualité de bout en bout est bornée par le
retrieval, pas par la génération.**

À lire avec précaution :

- **Un seul run, un seul juge.** Le juge est le même modèle que le générateur, ce
  qui tend à surestimer la faithfulness. Un score de 1.00 sur 44 réponses est
  plausible pour des réponses courtes extraites de passages précis, mais il n'a
  pas été validé par un test de calibration (par exemple soumettre au juge des
  réponses volontairement fausses et vérifier qu'il les signale). Ne pas le
  présenter comme une preuve d'absence d'hallucination.
- **Answer relevance calculée sur 39 réponses sur 44** : pour 5 d'entre elles, le
  juge a écrit le balisage d'appel d'outil dans un champ texte et laissé `score`
  vide (`output_errors = 5`, tous sur ce juge). Ces réponses sont donc exclues de
  la moyenne. Une nouvelle tentative automatique a été ajoutée depuis
  (`AnthropicLLMClient.complete_structured`) ; le run n'a pas été refait, donc
  ces chiffres ne la reflètent pas.
- Le jeu est de petite taille et en partie synthétique (voir plus haut) ;
  il sert à comparer des variantes et détecter des régressions, pas à annoncer
  un niveau absolu.
- L'ancien indicateur `abstention_accuracy` (0.83 dans ce run) mélange échecs de
  retrieval et refus injustifiés ; lisez plutôt `unanswerable_abstention_rate` et
  `false_abstention_rate`.

Deux formes de sortie mal formée du modèle ont été observées en conditions
réelles, que les tests avec doublures ne pouvaient pas montrer : un champ liste
renvoyé sous forme de chaîne JSON (réparé par `validate_tool_input`), et du
balisage d'appel d'outil fuyant dans un champ texte (retenté une fois). Une
sortie qui reste inexploitable est consignée par question (`output_errors`) sans
interrompre l'évaluation, alors qu'une erreur d'API (clé, crédit, réseau)
l'interrompt.

Un garde-fou de régression (`hit_rate >= 0.78`, `MRR >= 0.64` à k=5, mode
hybride) et un test « le hybride n'est pas pire que le dense » s'exécutent dans
la suite de tests, donc en CI.

### Suivi des runs (MLflow)

Chaque exécution de `python -m document_intelligence.evaluation` est aussi
enregistrée comme un run MLflow : la config (méthode de retrieval, modèle du
LLM-juge, `k`, taille du jeu de test) en paramètres, les métriques agrégées
(hit rate, MRR, faithfulness, citation precision...) en métriques, et le
rapport JSON détaillé des 63 questions en artefact. Le tracking est **local**
(`file:./mlruns`, pas de serveur dédié) ; `mlflow` est un groupe de
dépendances optionnel (`uv sync --group mlflow` ou `--all-groups`), et son
absence ne bloque ni l'évaluation ni la CI — le run est alors simplement
consigné comme non tracé.

Pour consulter les runs :

```bash
uv run mlflow ui --backend-store-uri file:./mlruns
```

### Optimisation des coûts LLM (model tiering + cache)

Le pipeline fait trois types d'appels à l'API Anthropic, tous via
`AnthropicLLMClient` : la **synthèse** de la réponse RAG (un appel par
question), et le **LLM-juge** d'évaluation (faithfulness, answer relevance).
Le routage est une règle Python, sans appel LLM.

- **Model tiering** (`orchestration/tiering.py`) : un modèle par rôle,
  configurable par variable d'environnement. `LLM_SYNTHESIS_MODEL` et
  `LLM_JUDGE_MODEL` valent `ANTHROPIC_MODEL` s'ils sont vides (comportement
  inchangé par défaut) ; `LLM_LIGHT_MODEL` (Claude Haiku par défaut) est réservé
  aux étapes simples (reformulation, classification, extraction de
  métadonnées), qui n'existent pas encore dans le pipeline.
- **Cache exact-match** (`orchestration/cache.py`) : clé = modèle + prompt
  système + prompt **normalisé** (Unicode NFKC, casse, espaces) + schéma de
  sortie. Le prompt contient les extraits récupérés : une réponse n'est jamais
  resservie pour un contexte différent. TTL (`LLM_CACHE_TTL_SECONDS`, 1 h),
  taille bornée en LRU (`LLM_CACHE_MAX_ENTRIES`), en mémoire ou dans un fichier
  JSON local (`LLM_CACHE_PATH`). Seules les sorties valides sont mises en cache.
- **Coût par run** (`orchestration/cost.py`) : chaque appel est journalisé avec
  son rôle et son modèle. Le rapport JSON (`cost`) et le run MLflow reçoivent
  `cost_actual_usd` (tiering + cache), `cost_unoptimized_usd` (tous les appels,
  cache compris, au tarif de `ANTHROPIC_MODEL`) et `cost_savings_pct`, au total
  et par rôle (`_synthesis`, `_judge`).

```bash
# Synthèse sur Haiku, juge sur Sonnet
LLM_SYNTHESIS_MODEL=claude-haiku-4-5 uv run python -m document_intelligence.evaluation --output rapport.json
```

**Mesure : synthèse Sonnet vs Haiku** (63 questions, k=5, mode hybride, juge
`claude-sonnet-5` dans les deux cas, un run chacun) :

| | Tout `claude-sonnet-5` | Synthèse `claude-haiku-4-5` | Écart |
|---|---|---|---|
| coût synthèse (mesuré) | 0.390 $ | 0.173 $ | **−55.6 %** |
| coût total du run (synthèse + juge) | 0.753 $ | 0.536 $ | −28.9 % |
| faithfulness | 0.977 (44 réponses) | 0.994 (45) | +0.017 |
| citation precision | 0.966 | 0.967 | = |
| answer relevance | 0.835 | 0.822 | −0.013 |
| questions sans réponse refusées | 8 / 8 | 8 / 8 | = |
| refus injustifiés | 2 / 46 | 1 / 46 | −1 |
| latence moyenne (hors juge) | 3.8 s | 2.4 s | −37 % |
| sorties invalides (`output_errors`) | 0 | 0 | = |

**Aucune dégradation mesurable de la qualité** : les écarts de faithfulness et
de relevance restent sous la variation observée d'un run Sonnet à l'autre
(relevance 0.90 lors du run précédent, 0.835 ici ; faithfulness 1.00 puis
0.977). En production, seule la synthèse appelle le LLM, donc **passer la
synthèse sur Haiku divise le coût LLM par ~2.25 (−55.6 %)**. Le défaut reste
Sonnet ; basculer se fait avec `LLM_SYNTHESIS_MODEL=claude-haiku-4-5`.

À lire avec précaution :

- Un run par configuration, 63 questions : suffisant pour écarter une
  régression nette, pas pour détecter un écart de quelques points.
- Le gain *estimé* par le journal pendant le run Haiku (50 %) est inférieur au
  gain *mesuré* (55.6 %). L'estimation valorise au tarif Sonnet les tokens
  comptés par Haiku, alors que la synthèse Sonnet a consommé ~13 % de tokens
  de plus pour les mêmes questions (tokenizers différents, réponses de longueur
  différente). Pour chiffrer un gain, comparez deux runs.
- **Le cache n'a rien économisé pendant l'évaluation** (0 hit) : les 63
  questions sont toutes distinctes. Son gain en production est égal à la part
  de requêtes répétées (même question, même contexte récupéré) dans la fenêtre
  du TTL ; il n'a pas été mesuré sur du trafic réel.
- Le juge reste Sonnet. Dans le run baseline il juge ses propres réponses, ce
  qui tend à le favoriser : le résultat Haiku n'est donc pas gonflé par ce biais.

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
