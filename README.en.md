# Document Intelligence Platform

[![CI](https://github.com/artichautttt/document-intelligence-platform/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/artichautttt/document-intelligence-platform/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

*English summary. The full documentation, with every measurement and its
caveats, is in French: [README.md](README.md). Every figure below is copied
from it; nothing here is measured separately.*

A retrieval-augmented generation (RAG) platform for querying business documents
(financial reports, contracts, audit reports). It answers **only from retrieved
passages, cites them, and refuses** when the documents do not contain the
answer. Around that core:

- an **MCP server** (official SDK) exposing search, cited answers and a data agent;
- interchangeable **LangGraph** and **LangChain** backends, and a native
  **tool-calling agent** (Anthropic Messages API);
- deterministic **guardrails** against prompt injection (disabled by default);
- a **text-to-SQL data agent** over a public dataset (NYC taxi trips);
- **evaluation** on a 63-question set tracked in **MLflow**, Prometheus/Grafana
  monitoring, Docker Compose, CI with `mypy --strict`.

The datasets used for evaluation were written by the author. They are meant to
compare variants and catch regressions, not to claim an absolute quality level.

## Architecture

```
client ──► api (FastAPI :8000)
             POST /documents : PDF/DOCX parsing → section chunking → local ONNX
                               embeddings → vector store (ChromaDB or Qdrant)
             POST /query     : LangGraph pipeline
                               route → retrieve (dense + BM25, RRF fusion)
                               → synthesize (LLM, structured output, citations)
                               → [optional CrewAI verification]
             GET /metrics/prometheus ──► prometheus ──► grafana
MCP clients (Claude Desktop, Claude Code) ──► mcp_server (stdio or HTTP)
                               same pipeline + SQL data agent
evaluation ──► ./mlruns (MLflow, local file store)
```

Only answer synthesis calls an LLM (Anthropic by default; OpenAI and Mistral
clients exist but were only tested against a simulated API).

## Run it

Requires Docker with Docker Compose.

```bash
cp .env.example .env     # optional: set ANTHROPIC_API_KEY to enable /query
docker compose up --build
```

API at http://localhost:8000 (interactive docs at `/docs`), Grafana at
http://localhost:3000, Prometheus at http://localhost:9090, MLflow UI with
`docker compose --profile mlflow up`. Without an API key, ingestion, search,
metrics and dashboards work; `/query` returns `503` with an explicit message.

Local development:

```bash
uv sync --all-groups
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
uv run python -m document_intelligence.mcp_server        # MCP server over stdio
```

## Measured results (and how far they go)

All runs below are **single runs** with `claude-sonnet-5`, k=5, hybrid
retrieval, on the 63-question set (55 answerable, 8 with no answer in the
documents). No statistical conclusion is drawn from them.

**Retrieval** (ChromaDB, local ONNX embeddings, 6 documents), hit rate / MRR /
nDCG at k=5: dense 0.76 / 0.64 / 0.67, hybrid 0.84 / 0.70 / 0.73. At k=5,
9 answerable questions out of 55 still get no relevant chunk. The lexical
re-ranking was designed after observing a failure on the first 15-question set,
so the sets are not independent of development.

**Generation** (LangGraph backend, judge = same model as the generator):

| Metric | Value | On |
|---|---|---|
| faithfulness | 1.00 | 44 judged answers |
| answer relevance | 0.90 | 39 answers (5 judge outputs unusable) |
| citation precision | 0.97 | answers citing at least one chunk |
| unanswerable questions refused | 8 / 8 | 8 |
| unjustified refusals (passage was in context) | 2 / 46 | well-retrieved answerable questions |
| mean latency (retrieval + synthesis) | 3.6 s | 63 |

The judge being the generator tends to overestimate faithfulness; the 1.00 was
not validated by a calibration test and is not evidence of zero hallucination.
End-to-end quality is bounded by retrieval, not generation.

**LangGraph vs LangChain backends** (same store, prompts and repair logic):
retrieval is identical by construction; faithfulness 0.992 vs 1.000, answer
relevance 0.938 vs 0.881, citation precision 0.966 vs 0.977, 8 / 8 refusals on
both, total cost 0.736 $ vs 0.722 $. The gaps come down to a few answers and are
within what one run of a non-deterministic model graded by an LLM judge can
produce: they are not attributed to the framework.

**Model tiering**: synthesis on `claude-haiku-4-5` instead of `claude-sonnet-5`
cost 0.173 $ instead of 0.390 $ for the 63 questions (−55.6 %), with
faithfulness 0.994 vs 0.977, citation precision 0.967 vs 0.966 and 8 / 8
refusals for both.

## MCP server

`src/document_intelligence/mcp_server/`, built with the official MCP SDK
(`FastMCP`), stdio or streamable HTTP. Six tools, all thin wrappers over
`tools.py` (no duplicated logic):

| Tool | What it does | LLM call |
|---|---|---|
| `search_documents(query, k)` | top-k chunks with score and source | no |
| `ask_question(question, k)` | cited answer from the configured RAG backend | yes |
| `list_documents()` | indexed documents and chunk counts | no |
| `describe_tables()` | documented schema of the data agent's database | no |
| `run_sql(sql)` | read-only `SELECT` behind the SQL guardrails | no |
| `ask_data(question)` | text-to-SQL agent, answer citing the executed query | yes |

```bash
claude mcp add document-intelligence -- uv run --directory /path/to/repo python -m document_intelligence.mcp_server
```

Tests (24) include an end-to-end test that starts the real server as a stdio
subprocess on a real ChromaDB; API keys are stripped, so CI makes no paid call.
One real `ask_question` round trip through MCP took 3.48 s (retrieval 0.86 s,
LLM 2.53 s) for 0.0063 $: a single call, proof that the chain works, not an
average.

## Guardrails (prompt injection, verified output)

Enabled with `GUARDRAILS_ENABLED=true`; **disabled by default**. No LLM call:

- **input**: 11 named regex rules applied to normalized, de-obfuscated text
  (accents, leetspeak, spaced letters). A matching question is refused; a
  matching chunk is quarantined at ingestion;
- **output**: each sentence must be supported by the passages it cites
  (numbers present, lexical overlap ≥ 0.5, no Markdown image, no foreign URL,
  no system-prompt leak). The answer passes, is degraded, or is refused.

Measured on the 63 questions and 40 attacks written by the author (one run):

- **4 / 55 unjustified refusals (7.3 %)**, above the 5 % target set for the
  project. This figure is optimistic: the leak-detector fix was written after
  seeing these questions. The 4 remaining refusals are limits of a lexical check
  (a derived number, paraphrases below the overlap threshold, an answer citing
  no passage). Unanswerable questions: 8 / 8 still refused.
- **Attacks detected by the rules: 24 / 24** for the subset committed before
  the rules, **26 / 40** overall, but **2 / 16** for attacks written after the
  rules (German, base64, synonyms...). Attacks and rules were written by the
  same author minutes apart; git history proves commit order, nothing more. The
  rules generalize poorly.
- Without guardrails the model executed none of the 40 attacks in this run
  (the canary string appeared in 8 answers, all of them refusals quoting it).
  The measurable benefit is early blocking and not returning attacker-controlled
  text, at the cost of the unjustified refusals above.

## Text-to-SQL data agent

Natural-language questions over a public, local SQLite database: NYC yellow taxi
trips for January 2025 (NYC TLC open data), aggregated by
`scripts/build_taxi_db.py` into a 2.1 MB database of 6 tables. Of 3,475,226 raw
trips, 3,330,486 are kept; dropped rows are counted per cleaning rule.

**SQL guardrails**, independent layers, each unit-tested on its own: lexical
checks (single statement, `SELECT`/`WITH` only, explicit refusal of write and
admin keywords, no recursive CTE), read-only connection (`mode=ro`), SQLite's
engine authorizer (`set_authorizer`: 6 allow-listed tables, allow-listed
functions including window functions, every other operation denied even if the
lexical checks were fooled), 200-row limit and 2 s timeout. Each refusal names
its rule.

**Agent**: native tool-calling loop (`run_sql`, `submit_answer`, forced tool
choice). The cited query is always one that was actually executed.

**Evaluation set**: 40 questions written by the author with Claude's help
(10 simple, 12 intermediate, 12 hard with multi-joins, window functions and
ratios, 6 unanswerable); expected values are the results of reference queries,
replayed in CI. A deterministic simulated client replays the whole set in CI and
a simulated "attacker" model tries 10 dangerous queries: this checks the
pipeline, guardrails and answer format, **not the quality of a real model**.

**Real measurement** (`claude-sonnet-5`, single run, interrupted):

> **29 of 40 questions were evaluated in the only real run, as the API credit
> ran out; 5 hard questions and the 6 unanswerable questions were not
> evaluated; no overall accuracy figure is therefore claimed; the set was
> written by the author with the schema in view.**

| Level | Evaluated | Correct value |
|---|---|---|
| simple | 10 / 10 | 10 / 10 |
| intermediate | 12 / 12 | 12 / 12 |
| hard | 7 / 12 | 7 / 7 asked |
| unanswerable | 0 / 6 | not evaluated |

On those 29 questions: no query blocked, no SQL error, mean latency 4.5 s,
cost 0.415 $. The MLflow run is kept and tagged as interrupted and not
representative. Known limits: on 2 questions the cited query is an arithmetic
query on numbers copied from an earlier result (correct value, weak
traceability); the guardrails were only triggered by tests and the simulated
attacker, never by the real model.

## Quality and CI

On 2026-10-25: `pytest` 503 passed, 1 skipped, 90 % coverage. GitHub Actions
runs `ruff check`, `ruff format --check`, `mypy --strict`, the tests on Python
3.11 and 3.12, and a Docker build with an API smoke test (no API key), on every
push or pull request to `main`.

## Repository layout

```
src/document_intelligence/
  api/            FastAPI endpoints
  mcp_server/     MCP server (6 tools)
  ingestion/      PDF and DOCX parsing
  chunking/       section-based chunking
  vectorization/  ChromaDB, Qdrant, hybrid dense + BM25
  orchestration/  LangGraph graph, LangChain chain, tool-calling agents, LLM clients, cost
  guardrails/     injection rules, ingestion quarantine, output verifier
  data_agent/     NYC taxi SQLite database, SQL guardrails, documented schema
  evaluation/     RAG and text-to-SQL evaluation, datasets (golden/), MLflow
  observability/  tracing, metrics, OpenTelemetry
tests/            pytest suite, test documents in fixtures/
scripts/          demos, judge reliability, taxi database build
finetune/         LoRA fine-tuning experiment
monitoring/       Prometheus and Grafana provisioning
```

## Out of scope

Real OpenAI and Mistral requests (clients tested on a simulated API only), a
hosted public demo, multi-user authentication (a single shared API key is
supported), and a complete real evaluation of the data agent.

**Demo coming later**: no screenshot or video is published yet, since it would
require paid API calls that are not being made at the moment. An AWS hosting
plan (not deployed, no resource created) is in
[docs/deploy/aws-plan.md](docs/deploy/aws-plan.md) (in French).

## License

MIT, see [LICENSE](LICENSE).
