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

    api_key: str | None = Field(default=None)


settings = Settings()
