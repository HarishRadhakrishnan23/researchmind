from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    openai_api_key: str
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Reranker
    cohere_api_key: str = ""

    # Vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "researchmind"

    # Security
    api_secret_key: str = "change-me"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    max_upload_size_mb: int = 50

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    # Agent
    agent_max_steps: int = 10

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — safe to call anywhere via Depends(get_settings)."""
    return Settings()
