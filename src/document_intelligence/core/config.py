"""Configuration centralisée de l'application, chargée depuis l'environnement."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres globaux de la plateforme, surchargeables via variables d'environnement."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    log_level: str = Field(default="INFO")
    max_chunk_chars: int = Field(default=1500, gt=0)
    chroma_persist_directory: str = Field(default=".chroma")

    vector_store_backend: str = Field(default="chroma")
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="document_chunks")

    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-sonnet-5")

    # Tarifs en USD par million de tokens, utilisés pour estimer le coût des appels LLM.
    # Volontairement à 0 par défaut : à renseigner d'après la grille tarifaire en vigueur.
    llm_input_price_per_mtok: float = Field(default=0.0, ge=0)
    llm_output_price_per_mtok: float = Field(default=0.0, ge=0)

    api_key: str | None = Field(default=None)

    # Persistance des métriques (GET /metrics) : fichier SQLite ; désactivée si vide.
    metrics_db_path: str | None = Field(default=None)

    # Export OpenTelemetry (Langfuse, Jaeger...) : activé si l'endpoint est défini.
    # Nécessite le groupe de dépendances `otel`.
    otel_exporter_otlp_endpoint: str | None = Field(default=None)
    otel_exporter_otlp_headers: str | None = Field(default=None)

    # "hybrid" : recherche dense + réordonnancement lexical ; "dense" : dense seul.
    retrieval_mode: str = Field(default="hybrid")


settings = Settings()
