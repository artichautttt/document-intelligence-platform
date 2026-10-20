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

    # Fournisseur LLM actif : "anthropic" (par défaut), "openai" ou "mistral".
    llm_provider: str = Field(default="anthropic")

    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-sonnet-5")
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4.1-mini")
    mistral_api_key: str | None = Field(default=None)
    mistral_model: str = Field(default="mistral-small-latest")

    # Model tiering : modèle par rôle d'appel LLM. Vide = `anthropic_model`, qui
    # sert aussi de référence « sans optimisation » pour chiffrer les économies.
    llm_synthesis_model: str | None = Field(default=None)
    llm_judge_model: str | None = Field(default=None)
    llm_verifier_model: str | None = Field(default=None)
    # Étapes simples (reformulation, classification, extraction de métadonnées).
    llm_light_model: str = Field(default="claude-haiku-4-5")
    # Juge de l'évaluation par rubrique (`evaluation.rubric_judge`), Anthropic
    # uniquement. Vide = modèle du LLM-juge (`LLM_JUDGE_MODEL`, sinon la référence).
    rubric_judge_model: str | None = Field(default=None)

    # Cache exact-match des réponses LLM (clé = prompt normalisé + modèle).
    # `llm_cache_path` vide = cache en mémoire seule ; sinon fichier JSON local.
    llm_cache_enabled: bool = Field(default=True)
    llm_cache_ttl_seconds: float = Field(default=3600.0, gt=0)
    llm_cache_max_entries: int = Field(default=1024, gt=0)
    llm_cache_path: str | None = Field(default=None)

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

    # Agent de vérification des réponses : "none" (défaut) ou "crewai"
    # (groupe de dépendances `agents`). Régénérations max après un verdict négatif.
    answer_verification: str = Field(default="none")
    verification_max_regenerations: int = Field(default=1, ge=0)

    # "hybrid" : recherche dense + réordonnancement lexical ; "dense" : dense seul.
    retrieval_mode: str = Field(default="hybrid")


settings = Settings()
