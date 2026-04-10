# =============================================================================
# app/core/config.py — Application Settings
# =============================================================================
# WHY THIS FILE EXISTS:
#   Every piece of config (DB URLs, secrets, model names) lives in .env.
#   Without this file you'd write os.environ.get("POSTGRES_URL") everywhere
#   — error-prone, no type checking, no defaults, no IDE autocomplete.
#
#   pydantic-settings reads the .env file and maps every variable into a
#   typed Python class. Benefits:
#   1. App crashes AT STARTUP if a required variable is missing
#      (not silently mid-request when a user is waiting)
#   2. Full type validation — EXPIRE_MINUTES="abc" raises an error immediately
#   3. IDE autocomplete — settings.jwt_secret, not magic strings everywhere
#   4. Single source of truth — one import, used everywhere
#
# USAGE ANYWHERE IN THE APP:
#   from app.core.config import settings
#   print(settings.postgres_url)
# =============================================================================

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # -------------------------------------------------------------------------
    # MODEL CONFIG
    # -------------------------------------------------------------------------
    # model_config tells pydantic-settings WHERE to read values from.
    # env_file=".env" means read from the .env file in the working directory.
    # case_sensitive=False means POSTGRES_URL and postgres_url both work.
    # extra="ignore" means unknown variables in .env don't cause errors —
    # useful when you add new env vars without updating this class yet.
    # -------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    # -------------------------------------------------------------------------
    # APPLICATION
    # -------------------------------------------------------------------------
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)

    # -------------------------------------------------------------------------
    # POSTGRESQL — static relational data
    # -------------------------------------------------------------------------
    # WHY TWO POSTGRES SETTINGS?
    #   database_url is the full async URL used by SQLAlchemy.
    #   The individual fields (postgres_user etc.) are read by Docker healthcheck
    #   and available for building URLs programmatically if needed.
    # -------------------------------------------------------------------------
    postgres_user: str = Field(default="friendagent")
    postgres_password: str = Field(default="friendagent_secret")
    postgres_db: str = Field(default="friendagent")
    database_url: str = Field(
        default="postgresql+asyncpg://friendagent:friendagent_secret@postgres:5432/friendagent"
    )

    # -------------------------------------------------------------------------
    # MONGODB — dynamic document data
    # -------------------------------------------------------------------------
    mongo_url: str = Field(
        default="mongodb://friendagent:mongo_secret@mongo:27017/friendagent?authSource=admin"
    )
    mongo_db: str = Field(default="friendagent")

    # -------------------------------------------------------------------------
    # REDIS — cache + celery broker
    # -------------------------------------------------------------------------
    redis_url: str = Field(default="redis://redis:6379/0")

    # -------------------------------------------------------------------------
    # QDRANT — vector store
    # -------------------------------------------------------------------------
    qdrant_url: str = Field(default="http://qdrant:6333")

    # -------------------------------------------------------------------------
    # OLLAMA — local LLM
    # -------------------------------------------------------------------------
    ollama_base_url: str = Field(default="http://ollama:11434")
    ollama_model: str = Field(default="llama3.2")
    ollama_embed_model: str = Field(default="nomic-embed-text")

    # -------------------------------------------------------------------------
    # LANGFUSE — LLM observability
    # -------------------------------------------------------------------------
    langfuse_public_key: str = Field(default="pk-lf-placeholder")
    langfuse_secret_key: str = Field(default="sk-lf-placeholder")
    langfuse_host: str = Field(default="http://langfuse:3000")

    # -------------------------------------------------------------------------
    # JWT AUTH
    # -------------------------------------------------------------------------
    # WHY JWT_SECRET MATTERS:
    #   This string is used to cryptographically sign every auth token.
    #   If someone knows this secret they can forge tokens for any user.
    #   In production: use a 64+ char random string, rotate it periodically.
    #   In development: the default is fine since it never leaves your machine.
    # -------------------------------------------------------------------------
    jwt_secret: str = Field(default="change_this_to_a_long_random_string_in_production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=10080)  # 7 days

    # LLM PROVIDER
    # Change this to switch providers: "gemini" | "anthropic" | "openai" | "ollama"
    llm_provider: str = Field(default="ollama")
    anthropic_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")


# =============================================================================
# WHY lru_cache?
#   Settings() reads the .env file from disk every time it's called.
#   lru_cache ensures this only happens ONCE — the result is cached in memory.
#   Every import of `settings` across the entire app gets the same object.
#   This is the standard FastAPI pattern for settings management.
# =============================================================================
@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Module-level singleton — import this everywhere
settings = get_settings()