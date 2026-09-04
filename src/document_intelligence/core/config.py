"""Configuration centralisée de l'application, chargée depuis l'environnement."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres globaux de la plateforme, surchargeables via variables d'environnement."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = Field(default="INFO")
    max_chunk_chars: int = Field(default=1500, gt=0)


settings = Settings()
