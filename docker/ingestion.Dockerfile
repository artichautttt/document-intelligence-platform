# Image minimale pour exécuter le pipeline d'ingestion + chunking (Sprint 1).
# Le multi-service docker-compose (vectorisation, API, agents) arrive au Sprint 2.
FROM python:3.11-slim

# libmagic + poppler-utils : dépendances système requises par `unstructured`
# pour la détection de type de fichier et le parsing PDF.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache .

ENV PYTHONPATH=/app/src

CMD ["python", "-c", "print('Pipeline ingestion + chunking pret. Utiliser en tant que librairie.')"]
