"""API settings, loaded from environment / .env (typed via pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "delegation-lab API"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor."""
    return Settings()
