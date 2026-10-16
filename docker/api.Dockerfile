# Image de l'API FastAPI : ingestion, chunking, vectorisation et
# recherche/synthese citee exposees via HTTP.
#
# Deux etages : le premier installe les dependances figees par uv.lock
# (--frozen, sans les groupes dev/agents/mlflow/otel) dans un virtualenv ;
# le second ne recoit que ce virtualenv, sans uv ni cache de build.

FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /bin/uv
# Pas de UV_COMPILE_BYTECODE : les .pyc precompiles ajoutaient ~190 Mo a l'image.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependances d'abord (couche mise en cache tant que uv.lock ne change pas).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.11-slim

# libmagic + poppler-utils : dependances systeme requises par `unstructured`
# pour la detection de type de fichier et le parsing PDF.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non root ; /data accueille la persistance ChromaDB (volume).
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/.chroma \
    && chown -R app:app /data

COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

USER app
WORKDIR /home/app

# Pre-telecharge le modele d'embedding ONNX (all-MiniLM-L6-v2, ~90 Mo) dans
# ~/.cache/chroma : le premier upload ne depend plus du reseau. L'archive
# telechargee (~83 Mo) est supprimee une fois extraite.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warmup'])" \
    && rm ~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "document_intelligence.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
