"""Typed, validated configuration loaded from environment variables.

Concepts: pydantic-settings, SecretStr, Literal types, cross-field validation,
fail-fast startup (a misconfigured service should crash at boot, not at the
first request).
"""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    # --- LLM provider -------------------------------------------------------
    llm_provider: Literal["fake", "anthropic"] = "fake"
    anthropic_api_key: SecretStr | None = None  # SecretStr: never printed in logs/repr
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_base_url: str = "https://api.anthropic.com"

    # --- Resilience -----------------------------------------------------------
    llm_timeout_s: float = Field(default=30.0, gt=0)
    stream_idle_timeout_s: float = Field(default=15.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay_s: float = Field(default=0.5, ge=0)
    retry_max_delay_s: float = Field(default=8.0, ge=0)
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_reset_timeout_s: float = Field(default=30.0, gt=0)

    # --- Concurrency / backpressure -------------------------------------------
    llm_max_concurrency: int = Field(default=8, ge=1)
    limiter_acquire_timeout_s: float = Field(default=10.0, gt=0)
    batch_concurrency: int = Field(default=4, ge=1)

    # --- Structured output ----------------------------------------------------
    structured_max_repairs: int = Field(default=2, ge=0, le=5)

    # --- Cache ----------------------------------------------------------------
    cache_ttl_s: float = Field(default=300.0, gt=0)
    cache_max_size: int = Field(default=1024, ge=1)

    @model_validator(mode="after")
    def _require_key_for_real_provider(self) -> Self:
        if self.llm_provider == "anthropic" and self.anthropic_api_key is None:
            raise ValueError("APP_ANTHROPIC_API_KEY is required when APP_LLM_PROVIDER=anthropic")
        if self.retry_base_delay_s > self.retry_max_delay_s:
            raise ValueError("retry_base_delay_s must be <= retry_max_delay_s")
        return self


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton. lru_cache makes it cheap to call anywhere."""
    return Settings()
