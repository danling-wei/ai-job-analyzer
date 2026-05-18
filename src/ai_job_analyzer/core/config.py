"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Values are read from environment variables (case-insensitive) and from a
    local ``.env`` file if present. See ``.env.example`` for the full list.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- App ---------------------------------------------------------------
    app_name: str = "ai-job-analyzer"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # ---- API ---------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # ---- LLM ---------------------------------------------------------------
    # Which provider to use in the agent workflow. The corresponding API key
    # for hosted providers must be set, otherwise the agent will fall back to a
    # local mock extractor. `qwen_lora` loads a local/Hugging Face Qwen model
    # in-process; `qwen_service` calls the separate model_lab Qwen service.
    llm_provider: Literal[
        "openai", "anthropic", "ollama", "qwen_lora", "qwen_service", "mock"
    ] = "mock"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1

    # Optional low-frequency finalizer. When `LLM_PROVIDER=qwen_service`, Qwen
    # still does per-posting extraction; this provider can do final skill
    # canonicalization and narrative synthesis.
    finalizer_provider: Literal["qwen_service", "openai", "mock"] = "qwen_service"
    finalizer_model: str = "gpt-5.2"
    finalizer_temperature: float = 0.0

    openai_api_key: str | None = None
    openai_base_url: str | None = None  # for OpenAI-compatible endpoints
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # ---- Local Qwen LoRA extraction ---------------------------------------
    qwen_base_model: str = "Qwen/Qwen3-1.7B"
    qwen_adapter_path: str | None = None
    qwen_device_map: str = "auto"
    qwen_torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    qwen_max_new_tokens: int = 900
    qwen_extraction_task: Literal["full_analysis", "skills_only"] = "full_analysis"
    qwen_service_host: str = "127.0.0.1"
    qwen_service_port: int = 8010
    qwen_service_url: str = "http://127.0.0.1:8010"
    qwen_service_timeout_seconds: float = 180.0
    qwen_service_batch_size: int = 1
    qwen_service_max_concurrency: int = 1

    # ---- Scraping ----------------------------------------------------------
    # Max number of postings to fetch per source per query.
    scrape_max_per_source: int = 20
    # Per-request HTTP timeout (seconds).
    http_timeout_seconds: float = 20.0
    # Whether to use Playwright for JS-rendered sources.
    enable_playwright: bool = True
    # User-Agent rotation (very small built-in pool).
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    # Reliable API-backed job sources. Scrapers return [] when required keys
    # are missing so local demos can still run with the mock source.
    serpapi_api_key: str | None = None

    # ---- Storage -----------------------------------------------------------
    # Local cache dir for raw HTML / parsed postings.
    cache_dir: str = ".cache"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance."""
    return Settings()
