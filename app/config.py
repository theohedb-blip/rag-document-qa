"""Application settings, read from environment variables (or a local .env file)."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Providers. Embeddings always use OpenAI; answers can use OpenAI or Anthropic.
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: Literal["openai", "anthropic"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    anthropic_model: str = "claude-haiku-5-5"

    # Storage
    chroma_dir: str = "data/chroma"
    collection_name: str = "documents"

    # Retrieval
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    min_score: float = 0.30  # cosine similarity; tuned with eval/run_eval.py --sweep

    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
