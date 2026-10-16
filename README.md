# Document Intelligence Platform

[![CI](https://github.com/artichautttt/document-intelligence-platform/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/artichautttt/document-intelligence-platform/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Plateforme RAG multi-agents pour interroger des documents d'entreprise (rapports
financiers, contrats, audits) : API FastAPI, pipeline LangGraph, réponses
citées et vérifiables, évaluation suivie dans MLflow, conteneurisée avec
Docker Compose.

## Problème

Retrouver une information précise dans des PDF et DOCX d'entreprise (un
montant, une clause, une échéance) prend du temps, et un LLM interrogé seul
invente facilement une réponse plausible. La plateforme :

1. ingère les documents (PDF, DOCX), les découpe par section et les indexe
   (ChromaDB ou Qdrant, recherche hybride dense + BM25) ;
2. répond à une question **uniquement à partir des extraits retrouvés**, en
   citant les passages utilisés, et **refuse de répondre** quand les documents
   ne contiennent pas l'information ;
3. mesure ce comportement (faithfulness, précision des citations, refus,
   coût) sur un jeu de 63 questions, et le surveille en production
   (Prometheus + Grafana).

## Démo

<!-- Remplacer par la capture ou la vidéo (2 min), par exemple :
![Démo](docs/demo.gif)
ou [Vidéo de démonstration (2 min)](https://...) -->
> Emplacement réservé : capture d'écran ou vidéo de démonstration (2 min), à ajouter.

## Architecture

```
                 ┌──────────────────────── docker compose ────────────────────────┐
  client HTTP    │                                                                │
 (curl, /docs) ──┼─► api (FastAPI, :8000)                                         │
                 │     POST /documents : ingestion PDF/DOCX → chunking par section│
                 │                       → embeddings ONNX locaux → vector store  │
                 │     POST /query     : graphe LangGraph                         │
                 │                       route → retrieve (dense + BM25, RRF)     │
                 │                       → synthesize (LLM, sortie structurée,    │
                 │                         citations) → [verify CrewAI, optionnel]│
                 │     GET /metrics/prometheus                                    │
                 │        │                    │                                  │
                 │        │ vector store       │ appel LLM (si clé configurée)    │
                 │        ▼                    ▼                                  │
                 │   ChromaDB embarqué    API Anthropic / OpenAI / Mistral        │
                 │   (volume chroma_data)  (hors compose)                         │
                 │   ou qdrant (:6333)                                            │
                 │                                                                │
                 │   prometheus (:9090) ──scrape──► api                           │
                 │   grafana (:3000) ──► prometheus (dashboard « Pipeline RAG »)  │
                 │   mlflow (:5000, profil optionnel) ──► ./mlruns (évaluations)  │
                 └────────────────────────────────────────────────────────────────┘
```

## Lancement en une commande

Prérequis : Docker avec Docker Compose. Aucune installation Python n'est
nécessaire.

```bash
cp .env.example .env          # optionnel : y mettre ANTHROPIC_API_KEY pour activer /query
docker compose up --build
```

| Service | URL | Rôle |
|---|---|---|
| `api` | http://localhost:8000 (doc interactive : `/docs`) | API FastAPI |
| `qdrant` | http://localhost:6333 | vector store dédié (`VECTOR_STORE_BACKEND=qdrant`) |
| `prometheus` | http://localhost:9090 | collecte des métriques de l'API |
| `grafana` | http://localhost:3000 (admin / `GRAFANA_ADMIN_PASSWORD`, `admin` par défaut) | dashboard « Pipeline RAG » |
| `mlflow` (optionnel) | http://localhost:5000 | runs d'évaluation de `./mlruns` : `docker compose --profile mlflow up` |

### Ce qui marche sans clé API, et ce qui en exige une

Vérifié le 2026-10-16 avec `docker compose up --build` (Docker Desktop, Windows) :

| Fonction | Sans clé | Avec `ANTHROPIC_API_KEY` |
|---|---|---|
| démarrage des 4 services, `GET /health`, `/docs` | ✅ | ✅ |
| `POST /documents` (ingestion + indexation, ChromaDB ou Qdrant) | ✅ (modèle d'embedding inclus dans l'image) | ✅ |
| `POST /query` (réponse citée) | ❌ `503` « Aucun fournisseur LLM n'est configuré » | ✅ `200` |
| métriques Prometheus, cible scrapée, dashboard Grafana | ✅ | ✅ |
| UI MLflow (profil `mlflow`) | ✅ | ✅ |

Seule la synthèse de réponse appelle un LLM. Avec `LLM_PROVIDER=openai` ou
`mistral`, c'est la clé du fournisseur choisi qui est requise (ces deux
fournisseurs n'ont été testés que sur API simulée, voir plus bas).

### Exemples d'appel

Réponses réelles obtenues pendant la vérification, avec les documents de test
de `tests/fixtures/documents/` :

```bash
curl -F "file=@tests/fixtures/documents/financial_report.pdf" http://localhost:8000/documents
# 201 {"document_id":"4e8ccef8-dbdb-4dca-b7c7-d0125bfa8abe","chunk_count":6}

curl -H "Content-Type: application/json" \
     -d '{"query": "Quel est le chiffre d affaires du trimestre ?", "k": 5}' \
     http://localhost:8000/query
# sans clé : 503 {"detail":"Aucun fournisseur LLM n'est configuré sur ce déploiement."}
# avec clé (claude-sonnet-5, 2.9 s, 2 285 tokens d'entrée / 228 de sortie) : 200
# {"query": "...", "answer": "Les chiffres d'affaires trimestriels sont les suivants :
#   Q1 = 10 800 k€ (marge 17,1%), Q2 = 11 950 k€ (marge 18,0%), Q3 = 12 400 k€
#   (marge 18,9%), Q4 = 13 050 k€ (marge 19,5%).",
#  "citations": [1 extrait cité], "sufficient_context": true, ...}
# (la réponse contient aussi les champs "verified" et "regenerations")
```

Si `API_KEY` est défini, `/documents`, `/query` et `/metrics` exigent l'en-tête
`X-API-Key: <valeur>`. `/health` reste public.

## Résultats

Uniquement des chiffres présents dans les rapports d'évaluation et les runs
MLflow du dépôt de développement (`rapport_*.json` et `mlruns/`, non
versionnés car locaux). 63 questions, k=5, retrieval hybride, juge
`claude-sonnet-5`. Un run par configuration.

**Fine-tuning LoRA de la synthèse** (Qwen2.5-0.5B-Instruct, CPU) :

| Métrique | Qwen 0.5B base | Qwen 0.5B + LoRA |
|---|---|---|
| faithfulness | 0,29 | **0,84** |
| citation precision | 0 | **0,87** |
| refus sur questions sans réponse | 0 / 8 | **7 / 8** |
| refus injustifiés (passage pourtant retrouvé) | 0,04 | 0,37 (dégradation) |

Sources : `rapport_ft_base.json` / run MLflow `ec0f1756` ; `rapport_ft_lora.json`
/ run MLflow `602925de`. Un premier run LoRA sur les mêmes 63 questions
(`a88e4f12`) avait donné faithfulness 0,89 et citation precision 0,73 ; le
tableau reprend le dernier run, celui dont le rapport est conservé.

**Coût de la synthèse : Sonnet vs Haiku** :

| | `claude-sonnet-5` | `claude-haiku-4-5` |
|---|---|---|
| coût de synthèse (63 questions) | 0,390 $ | **0,173 $** (−55,6 %) |
| faithfulness | 0,977 | 0,994 |
| citation precision | 0,966 | 0,967 |
| refus sur questions sans réponse | 8 / 8 | 8 / 8 |

Sources : runs MLflow `49b36005` (Sonnet) et `184ffba6` (Haiku), métrique
`cost_actual_usd_synthesis`.

Détails, limites et autres mesures (retrieval, agent de vérification) : voir
*Évaluation*, *Agent de vérification* et *Fine-tuning LoRA* plus bas.

## Qualité

Mesuré le 2026-10-16 (Python 3.11, Windows) :

- `pytest --collect-only -q` : **187 tests** ;
- `pytest --cov=document_intelligence` : **186 passés, 1 ignoré**, couverture
  **90 %** (1 965 instructions, 187 non couvertes) ;
- `ruff check`, `ruff format --check`, `mypy` (strict) : exécutés en CI, voir *CI/CD*.

```bash
uv sync --all-groups
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov=document_intelligence
```

## Structure du dépôt

```
.
├── src/document_intelligence/
│   ├── api/             # FastAPI : /documents, /query, /metrics, /health
│   ├── ingestion/       # parsing PDF (unstructured + pdfplumber) et DOCX (unstructured)
│   ├── chunking/        # découpage par section
│   ├── vectorization/   # VectorStore : ChromaDB, Qdrant, hybride dense + BM25
│   ├── orchestration/   # graphe LangGraph, clients LLM, tiering, cache, coût, vérificateur CrewAI
│   ├── evaluation/      # évaluation RAG, jeu de 63 questions (golden/), suivi MLflow
│   ├── observability/   # tracing, métriques, Prometheus, OpenTelemetry
│   └── core/            # configuration (variables d'environnement), logs
├── tests/               # 187 tests pytest, documents de test dans fixtures/
├── finetune/            # expérience LoRA : jeu de distillation et entraînement
├── monitoring/          # config Prometheus, provisioning et dashboard Grafana
├── docker/              # Dockerfiles de l'API et de l'ingestion
├── docker-compose.yml   # api, qdrant, prometheus, grafana (+ mlflow en profil)
├── .env.example         # toutes les variables lues par le code, sans valeur secrète
└── .github/workflows/   # CI : lint, mypy strict, tests 3.11/3.12, build Docker
```

## Développement local (sans Docker)

```bash
uv sync                  # cœur + outils de dev
uv sync --all-groups     # + mlflow, otel, agents (CrewAI)
uv run uvicorn document_intelligence.api.app:app --reload
```

Utilisation en bibliothèque :

```python
from document_intelligence.ingestion.service import ingest
from document_intelligence.chunking.service import chunk_document
from document_intelligence.vectorization.chroma_store import ChromaStore
from document_intelligence.orchestration.anthropic_client import AnthropicLLMClient
from document_intelligence.orchestration.graph import answer_query

store = ChromaStore()
store.add_chunks(chunk_document(ingest("rapport.pdf")))
answer = answer_query(
    "Quels sont les resultats financiers du trimestre ?",
    store=store,
    llm=AnthropicLLMClient(api_key="sk-ant-..."),
)
print(answer.answer, [c.chunk_id for c in answer.citations])
```

---

# Détails techniques

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

## Fournisseurs LLM (Anthropic, OpenAI, Mistral)

`LLM_PROVIDER=anthropic|openai|mistral` (défaut `anthropic`) choisit le client
construit par `orchestration/tiering.build_llm_client`, pour l'API comme pour
l'évaluation. Chaque fournisseur a sa clé et son modèle de référence :

| `LLM_PROVIDER` | Client | Clé | Modèle par défaut |
|---|---|---|---|
| `anthropic` | `AnthropicLLMClient` (API Messages) | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` = `claude-sonnet-5` |
| `openai` | `OpenAILLMClient` (Chat Completions) | `OPENAI_API_KEY` | `OPENAI_MODEL` = `gpt-4.1-mini` |
| `mistral` | `MistralLLMClient` (Chat Completion) | `MISTRAL_API_KEY` | `MISTRAL_MODEL` = `mistral-small-latest` |

- **Socle commun** (`orchestration/instrumented.py`) : cache exact-match,
  journal de coût, tracing `llm.call`, métriques Prometheus et nouvelle
  tentative sur sortie invalide sont écrits une fois. Un fournisseur
  n'implémente que deux appels bruts : complétion texte et appel d'outil forcé.
- **Sorties structurées** : appel de fonction forcé (`tool_choice` nommé) chez
  les trois fournisseurs. OpenAI n'utilise pas `response_format` strict, qui
  exige que tous les champs soient requis.
- **Tiering** : `LLM_SYNTHESIS_MODEL`, `LLM_JUDGE_MODEL` et `LLM_VERIFIER_MODEL`
  s'appliquent au fournisseur actif. `LLM_LIGHT_MODEL` (Claude Haiku) ne vaut
  qu'avec Anthropic ; ailleurs, le rôle léger prend le modèle de référence.
- **Tarifs** (`orchestration/cost.py`) : `gpt-4.1-mini` 0.40 / 1.60 $ et
  `mistral-small-latest` 0.15 / 0.60 $ par million de tokens, relevés sur des
  agrégateurs de prix en octobre 2026, **pas sur les grilles officielles**.
  `mistral-small-latest` est un alias : le modèle visé, donc son prix, peut
  changer (certaines sources donnent 0.10 / 0.30 $).

Ce qui a été vérifié, et comment :

| Fournisseur | Vérification | Résultat |
|---|---|---|
| Anthropic | **Requête réelle** à travers le pipeline (3 fixtures, retrieval hybride, synthèse structurée), client construit par `LLM_PROVIDER=anthropic` | réponse correcte et citée ; 2 274 tokens d'entrée / 168 de sortie ; 2.93 s ; 0.006228 $ |
| OpenAI | **API simulée** (`httpx.MockTransport`) : le vrai SDK `openai` envoie la requête et parse une réponse au format documenté | 7 tests + 1 test de pipeline RAG complet (ingestion, ChromaDB, LangGraph) |
| Mistral | **API simulée** (`httpx2.MockTransport`) avec le vrai SDK `mistralai` | mêmes 8 tests |

**OpenAI et Mistral n'ont jamais été appelés en conditions réelles** (pas de
clé disponible). Les tests prouvent l'intégration avec les SDK : format de
requête, appel d'outil forcé, parsing, réparation et nouvelle tentative,
erreurs d'API, tokens et coût. Ils ne prouvent ni que les noms de modèles par
défaut sont acceptés par les API, ni la qualité des réponses de ces modèles,
ni leur respect de l'appel d'outil forcé. Avant de les citer comme
« fonctionnels », il faut une requête réelle (`LLM_PROVIDER=openai` +
`OPENAI_API_KEY`, puis `python -m document_intelligence.evaluation --sample 5`).

Effet de bord du lock `uv` unifié : CrewAI (groupe `agents`) contraint aussi
le cœur. `openai` est plafonné à 2.54 (3.x existe) et `pydantic` à 2.12.5 au
lieu de 2.13.5.

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

## Agent de vérification (CrewAI)

Après la synthèse, un agent CrewAI « vérificateur de faits » reçoit la
question, la réponse et les extraits sources. Il rend un verdict structuré :
`grounded`, `unsupported_claims`, `feedback`. Si la réponse n'est pas fondée,
le graphe la régénère en transmettant au rédacteur le retour du vérificateur,
dans la limite de `VERIFICATION_MAX_REGENERATIONS` (1 par défaut). Au-delà, la
réponse est rendue avec `verified=false`.

```
route → retrieve → synthesize → verify ─┬─ grounded / limite atteinte → fin
                       ↑                └─ non fondée → synthesize (avec retour)
```

```bash
uv sync --group agents
ANSWER_VERIFICATION=crewai uv run uvicorn document_intelligence.api.app:app
uv run python -m document_intelligence.evaluation --sample 30 --verify --calibrate-verifier
```

**Pourquoi CrewAI plutôt qu'AutoGen** avec ce graphe LangGraph :

- `Crew.kickoff()` est synchrone et s'appelle directement depuis un nœud
  LangGraph synchrone. AutoGen ≥ 0.4 (`autogen-agentchat`) n'a qu'une API
  async : il faudrait un `asyncio.run()` dans le nœud, qui échoue quand une
  boucle tourne déjà, ce qui est le cas sous FastAPI (`/query`).
- CrewAI accepte un LLM personnalisé (`BaseLLM`). `LLMClientAdapter`
  (`orchestration/crew_verifier.py`) délègue au `LLMClient` du projet, donc
  le vérificateur suit le fournisseur actif, le tiering (`LLM_VERIFIER_MODEL`),
  le cache, le journal de coût et les métriques. Quand CrewAI demande une
  sortie Pydantic, l'adaptateur passe par `complete_structured` (function
  calling) : un seul appel LLM par vérification, observé dans les tests.
- La boucle de régénération reste dans LangGraph (arête conditionnelle) ;
  CrewAI ne fait que le jugement.

Contrat `AnswerVerifier` (`orchestration/verification.py`) : une autre
implémentation (doublure, autre framework) se branche sans toucher au graphe.
Une abstention n'est pas vérifiée (rien à contrôler). Un verdict inexploitable
laisse la réponse non vérifiée (`verified=null`) sans casser la requête. Une
erreur d'API est propagée. Une régénération change le prompt, donc le cache ne
ressert jamais la réponse rejetée. `/query` expose `verified` et
`regenerations`.

### Mesure (`claude-sonnet-5` partout, 30 questions tirées au hasard, graine 0, k=5, hybride)

Un run, le 2026-10-07 : `rapport_verification30.json`, non versionné. Les 30
questions comptent 28 répondables et 2 sans réponse.

| | Valeur |
|---|---|
| réponses vérifiées (non abstenues) | 22 / 30 (8 abstentions, non vérifiées) |
| **réponses réelles signalées par l'agent** | **0 / 22** |
| régénérations déclenchées | 0 |
| **calibration : copies faussées détectées** | **17 / 17** (5 réponses sans nombre, non testées) |
| sorties invalides (`output_errors`) | 0 |
| latence du nœud `verify` (moyenne, n=22) | 2.2 s (synthèse : 2.8 s ; pipeline complet : 5.1 s) |
| coût vérificateur | 0.271 $ pour 39 appels (≈ 0.0069 $ / appel ; synthèse ≈ 0.0063 $ / appel) |
| coût total du run (synthèse + vérif. + calibration + juge) | 0.636 $ |

Le juge (même run) donne faithfulness 0.989, answer relevance 0.920, citation
precision 0.932, 2 / 2 questions sans réponse refusées et 1 refus injustifié
sur 23 (0.043).

**Boucle de régénération avec les vrais modèles**. Comme aucune réponse réelle
n'a été signalée, la régénération a été testée à part : la première synthèse
de chaque question est volontairement faussée (premier nombre altéré), les
suivantes sont celles du vrai modèle. Script de vérification ponctuel, non
versionné, mêmes 30 questions :

| | Valeur |
|---|---|
| réponses faussées injectées | 17 |
| rejetées par le vérificateur | 17 / 17 |
| régénérées, puis validées au 2ᵉ passage | 17 / 17 |
| valeur faussée absente de la réponse finale | 17 / 17 (valeur d'origine présente dans les 17) |
| coût de l'expérience | 0.576 $ |

À lire avec précaution :

- **« 0 signalement » ne prouve pas l'absence d'hallucination.** Sur 22
  réponses, l'agent n'a rien signalé. Le juge, lui, a noté une réponse 0.75 en
  faithfulness : la durée de confidentialité, où la réponse cite deux articles
  de deux contrats différents. Le vérificateur l'a validée. Lequel a raison
  n'a pas été tranché à la main ; c'est un désaccord, pas un faux négatif
  établi.
- **La calibration ne teste que des substitutions de nombres**, l'erreur la
  plus facile à détecter. Selon la réponse, le nombre altéré est un montant,
  un pourcentage, un numéro d'article ou de trimestre (« Q3 » → « Q6 »). Les
  extrapolations sans chiffre, les ajouts plausibles et les erreurs de
  raisonnement ne sont pas mesurés. 17/17 est donc une borne haute
  optimiste, pas un taux de détection général.
- Le vérificateur est le même modèle que le générateur et que le juge : leurs
  angles morts peuvent être communs.
- Un seul run, 30 questions : suffisant pour montrer que le mécanisme
  fonctionne, pas pour estimer un taux d'hallucination.
- **Coût et latence** : la vérification coûte à peu près autant que la
  synthèse et ajoute ≈ 2.2 s par réponse. Avec `LLM_VERIFIER_MODEL` sur un
  modèle plus léger, l'économie possible n'a pas été mesurée.
- CrewAI est lent à importer : 36 s mesurées à froid, 12 s à chaud sur la
  machine de développement. Il n'est importé que si la vérification est
  activée. Il ajoute aussi son propre échafaudage de prompt (rôle, objectif,
  schéma OpenAPI), donc des tokens d'entrée.
- L'image Docker (`docker/api.Dockerfile`) n'installe pas le groupe `agents` :
  `ANSWER_VERIFICATION=crewai` n'y fonctionne pas en l'état.
- Mesures faites sous Python 3.14 en local ; la CI exécute les tests CrewAI
  sous 3.11 et 3.12 (`uv sync --all-groups`), ce qui n'a pas été vérifié ici.

## Fine-tuning LoRA de la synthèse (expérience)

Objectif : vérifier si un petit modèle open source, spécialisé par LoRA sur la
tâche de synthèse citée, peut remplacer Claude pour ce nœud. Le modèle choisi
est `Qwen/Qwen2.5-0.5B-Instruct`, entraîné et servi sur CPU via
`HFLocalLLMClient`. Le retrieval, le prompt et le schéma `GroundedAnswer` sont
inchangés.

```bash
pip install -r finetune/requirements.txt
uv run python finetune/build_dataset.py --out finetune/data          # distillation (API Claude)
python finetune/train_lora.py --data finetune/data --out finetune/adapter --epochs 1 --holdout-doc 9
python -m document_intelligence.evaluation --local-model Qwen/Qwen2.5-0.5B-Instruct \
    --local-adapter finetune/adapter/epoch1 --output rapport_ft_lora.json
```

### Jeu de données (distillation, corpus disjoint de l'évaluation)

- Claude rédige 10 documents fictifs (contrats, rapports financiers, audits).
  Il a l'interdiction d'utiliser les entités du corpus d'évaluation. Ces
  documents passent par la vraie chaîne ingestion → chunking → retrieval
  hybride (k=5).
- Claude rédige aussi les questions : 24 répondables et 6 sans réponse par
  document. Pour chaque question, il produit la réponse cible à partir des
  extraits réellement retrouvés, avec le même prompt système que le pipeline.
- Résultat : **316 exemples**, dont 256 répondables, 194 où le passage attendu
  figure dans le contexte et 102 cibles d'abstention. Coût de l'enseignant :
  2,25 $.
- **Contrôle de fuite** (`leakage_report.json`) :
  - aucune entité interdite n'apparaît dans le corpus d'entraînement ;
  - 3 formules génériques sont partagées avec les passages d'évaluation
    (« manquement grave », « délai de 48 heures », « tribunal de commerce de
    Lyon ») ;
  - 2 questions sont quasi identiques à des questions d'évaluation (ratio
    ≥ 0,8).
  - **9 exemples exclus** de l'entraînement : ces 2 questions, plus les cibles
    qui contenaient l'une des 3 formules.

### Entraînement (un seul run)

| Paramètre | Valeur |
|---|---|
| LoRA | rang 16, alpha 32, dropout 0,05, q/k/v/o/gate/up/down_proj |
| Paramètres entraînables | 8 798 208 / 502 830 976 (1,75 %) |
| Données | 277 train, 30 validation (document 9 entier mis de côté) |
| Optimisation | AdamW, lr 2e-4 linéaire vers 0, batch effectif 8, 1 époque, perte sur la réponse seule |
| Matériel / durée | CPU 4 cœurs, gradient checkpointing, **973,8 min** |
| val_loss | 0,7319 (avant) → **0,4573** ; train_loss 0,4750 |

### Résultat : 63 questions, même juge (`claude-sonnet-5`), même retrieval

| Métrique | Qwen 0.5B base | Qwen 0.5B + LoRA |
|---|---|---|
| faithfulness | 0,293 (58 réponses jugées) | **0,839** (31 réponses jugées) |
| answer_relevance | 0,116 | **0,726** |
| citation_precision | 0,000 | **0,867** |
| refus sur questions sans réponse | 0/8 | **7/8** |
| abstention_accuracy | 0,794 | 0,607 |
| refus injustifié (passage dans le contexte) | 0,043 | **0,370** (dégradation) |
| sorties hors schéma | 0 | 2 / 63 |
| latence moyenne (CPU) | 14,7 s | 17,9 s |

Le retrieval est identique dans les deux runs (hit rate 0,836, MRR 0,701), car
seule la synthèse change.

**Lecture honnête :**

- **Gains nets** : le modèle de base recopiait l'identifiant d'exemple du
  prompt et inventait des chiffres. Le modèle LoRA cite des extraits valides
  et refuse 7 fois sur 8 quand la réponse n'existe pas.
- **Dégradation** : il refuse à tort dans 37 % des cas où le passage attendu
  figure dans le contexte, contre 4 % pour le modèle de base. Une partie du gain de faithfulness vient
  donc de cette prudence excessive : il répond moins souvent (31 réponses
  jugées contre 58), mais mieux.
- **Toujours en dessous de Claude** sur la même évaluation : faithfulness
  1,00, citation precision 0,97, 8/8 refus justifiés et refus injustifié 0,04
  (voir *Génération mesurée*).
- **Limites de l'expérience** :
  - un seul run, une seule graine, une seule époque, sans recherche
    d'hyperparamètres ;
  - 63 questions d'évaluation ;
  - un juge LLM unique ;
  - un enseignant issu de la même famille que le juge (biais possible) ;
  - 102 cibles d'abstention sur 316 (32 %), ce qui explique probablement le
    biais vers le refus ;
  - aucun intervalle de confiance.

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

## Monitoring (Prometheus + Grafana)

Même pattern que le Mini-SIEM : `prometheus_client`, registre par défaut,
scrape par Prometheus, dashboard Grafana provisionné à 4 panels.

`GET /metrics/prometheus` (format texte ; `GET /metrics` reste l'instantané
JSON historique) expose :

| Métrique | Type | Contenu |
|---|---|---|
| `rag_queries_total{outcome}` | compteur | requêtes `/query` : `answered`, `refused` (`sufficient_context=false`), `error` |
| `rag_query_duration_seconds{outcome}` | histogramme | latence de bout en bout par requête |
| `rag_query_cost_usd` | histogramme | coût LLM **par requête** : appels faits pendant la requête, au tarif du modèle effectivement appelé (tiering compris), hit de cache = 0 |
| `rag_llm_cost_usd_total`, `rag_llm_tokens_total`, `rag_llm_calls_total` | compteurs | cumul par `provider`, `model`, `role` (et `direction` / `source`) |
| `rag_answer_verification_total{result}` | compteur | verdicts de l'agent : `grounded`, `flagged` |

Le coût par requête est accumulé dans une `ContextVar` ouverte autour de
`/query`. Les appels LLM du vérificateur et des régénérations comptent dans la
même requête.

```bash
docker compose up -d --build   # api, qdrant, prometheus (:9090), grafana (:3000)
```

Fichiers : `monitoring/prometheus/prometheus.yml` (job `rag-api`, cible
`api:8000/metrics/prometheus`), `monitoring/grafana/provisioning/`
(datasource et fournisseur de dashboards), `monitoring/grafana/dashboards/rag-pipeline.json`.
Dashboard « Pipeline RAG », 4 panels : requêtes/s par issue, latence p50/p95,
taux de refus, coût LLM moyen par requête.

**Testé et vérifié le 2026-10-07** (Docker Desktop, `docker compose up`,
synthèse sur `claude-haiku-4-5`) :

- Prometheus scrape l'API : cible `rag-api` en `health: up` dans `/api/v1/targets`.
- Grafana a provisionné la datasource Prometheus et le dashboard « Pipeline
  RAG » (`/api/datasources`, `/api/search`).
- Trafic de test, 2 documents ingérés : 4 questions répondues, 2 questions sans
  réponse dans les documents, 1 requête vide (400). Compteurs observés :
  `answered` 4, `refused` 2, `error` 1 ; 6 appels Haiku, 12 621 tokens
  d'entrée / 952 de sortie, 0.017381 $ cumulés (= 12 621 × 1 $ + 952 × 5 $ par
  million).
- Les requêtes des 4 panels, exécutées par Grafana (`/api/ds/query`) et par
  Prometheus, renvoient les mêmes valeurs : taux de refus 0.333 (2 / 6),
  coût moyen 0.002483 $ / requête, latence p50 2.5 s / p95 2.95 s.
  **Vérifié par API, pas visuellement dans l'interface Grafana.**

Bug préexistant corrigé à cette occasion : `docker-compose.yml` passe
`API_KEY=` (chaîne vide) quand la variable n'est pas définie. Or l'API ne
désactivait l'authentification que pour `None` : sous Docker, `/documents`,
`/query` et `/metrics` répondaient donc 401 sans en-tête. Une chaîne vide
vaut désormais « non configurée » (test `test_empty_api_key_disables_authentication`).

Limites :

- Le coût moyen par requête inclut les requêtes en erreur (coût souvent 0),
  ce qui peut le diluer. Le taux de refus exclut les erreurs.
- Les panels « taux de refus » et « coût moyen » sont cumulés depuis le
  démarrage de l'API, lisibles à faible trafic. En production, on préférerait
  une fenêtre glissante (`increase(...[1h])`).
- p50/p95 sont interpolés dans les buckets de l'histogramme (bornes 2 s, 3 s,
  5 s...) : ce sont des approximations, pas des percentiles exacts.
- Les requêtes rejetées avant le handler (JSON invalide, validation FastAPI)
  ne sont pas comptées : ce ne sont pas des requêtes RAG.
- Si `API_KEY` est défini, `/metrics/prometheus` exige `X-API-Key`. La config
  Prometheus fournie ne l'envoie pas : laisser `API_KEY` vide ou ajouter
  l'en-tête au scrape.
- Pas d'Alertmanager ni de règles
  d'alerte (contrairement au Mini-SIEM), compose seulement (pas de manifests K8s).
- Seules les requêtes de l'API sont mesurées : l'évaluation CLI n'alimente pas
  ces métriques de requête.

## Docker

```bash
docker compose up --build                     # api, qdrant, prometheus, grafana
docker compose --profile mlflow up -d mlflow  # + UI MLflow sur ./mlruns
```

`docker/api.Dockerfile` :

- **deux étages** : le premier installe les dépendances figées par `uv.lock`
  (`uv sync --frozen --no-dev`, sans les groupes `agents`, `mlflow`, `otel`) ;
  le second ne reçoit que le virtualenv, sans uv ni cache de build ;
- **utilisateur non root** (`app`, uid 1000) ;
- **modèle d'embedding inclus** (all-MiniLM-L6-v2 ONNX, ~90 Mo) : l'upload
  fonctionne sans accès réseau (vérifié avec `docker run --network none`) ;
- `HEALTHCHECK` sur `/health` ; `.dockerignore` exclut `.env`, les venvs, les
  caches, `mlruns/` et les rapports locaux du contexte de build.

Taille mesurée : 1,64 Go, contre 1,57 Go pour l'ancienne image à un étage.
L'écart vient du modèle désormais inclus (l'ancienne image le téléchargeait au
premier upload). Le gros du poids vient des dépendances de `unstructured` et
`chromadb` (spacy, llvmlite, kubernetes...).

Images des autres services épinglées : `qdrant/qdrant:v1.19.2`,
`prom/prometheus:v3.15.0`, `grafana/grafana:13.2.4`,
`ghcr.io/mlflow/mlflow:v3.16.1`. Par défaut `api` utilise ChromaDB embarqué
(volume `chroma_data`) ; `VECTOR_STORE_BACKEND=qdrant` bascule sur le service
`qdrant` sans changer le contrat `VectorStore` (vérifié : 6 chunks indexés
dans la collection `document_chunks`).

Limites :

- le groupe `agents` (CrewAI) n'est pas installé dans l'image :
  `ANSWER_VERIFICATION=crewai` n'y fonctionne pas en l'état ;
- l'UI MLflow lit les métriques et paramètres des runs ; les artefacts
  enregistrés avec un chemin absolu Windows ne sont pas lisibles depuis le
  conteneur ;
- `docker/ingestion.Dockerfile` (image de bibliothèque, sans service) n'a pas
  été modifié.

## CI/CD

`.github/workflows/ci.yml` s'exécute à chaque push/PR sur `main`, en trois jobs :

- **lint** — `ruff check`, `ruff format --check`, puis `mypy --strict` (0 erreur).
- **test** — `pytest` avec couverture, en matrice Python 3.11 et 3.12.
- **docker-build** — build des images `ingestion` et `api` via Buildx, avec cache
  de layers GitHub Actions ; ne démarre que si `lint` et `test` passent.

## Hors périmètre (à venir)

- Requêtes réelles OpenAI et Mistral (clients livrés, testés sur API
  simulée seulement) ; modèle local (Ollama...) via une implémentation
  `InstrumentedLLMClient` supplémentaire.
- Migration vers Pinecone (Qdrant est désormais disponible ; Pinecone resterait
  une implémentation `VectorStore` supplémentaire si besoin).
- Authentification par clé d'API unique et partagée : pas de gestion
  multi-utilisateurs, de rotation de clé ou de scopes ; à envisager si l'API
  est exposée à plusieurs clients distincts.

## Licence

MIT — voir [LICENSE](LICENSE).
