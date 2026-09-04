# Image de l'API FastAPI (Sprint 4) : ingestion, chunking, vectorisation et
# recherche/synthese citee exposees via HTTP.
FROM python:3.11-slim

# libmagic + poppler-utils : dependances systeme requises par `unstructured`
# pour la detection de type de fichier et le parsing PDF.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache .

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "document_intelligence.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
