"""Runtime configuration, loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (src/sift/config.py -> project/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Output dimension for embedding models we ship prefixes/support for.
KNOWN_EMBED_DIMS: dict[str, int] = {
    "baai/bge-large-en-v1.5": 1024,
    "baai/bge-base-en-v1.5": 768,
    "baai/bge-small-en-v1.5": 384,
    "snowflake/snowflake-arctic-embed-l": 1024,
    "snowflake/snowflake-arctic-embed-m": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "intfloat/multilingual-e5-large": 1024,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="",
        extra="ignore",
    )

    # --- paths ---
    vault_path: Path = Field(default=PROJECT_ROOT / "vault", alias="SIFT_VAULT_PATH")
    db_path: Path = Field(default=PROJECT_ROOT / "data" / "lancedb", alias="SIFT_DB_PATH")

    # --- embeddings ---
    embed_model: str = Field(default="BAAI/bge-base-en-v1.5", alias="SIFT_EMBED_MODEL")
    embed_device: str = Field(default="auto", alias="SIFT_EMBED_DEVICE")  # auto|cuda|cpu
    embed_dim: int = Field(default=0, alias="SIFT_EMBED_DIM")  # 0 = derive from model

    # --- retrieval ---
    rerank: bool = Field(default=False, alias="SIFT_RERANK")
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="SIFT_RERANK_MODEL")

    # --- external APIs ---
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    h1_api_username: str | None = Field(default=None, alias="H1_API_USERNAME")
    h1_api_token: str | None = Field(default=None, alias="H1_API_TOKEN")
    nvd_api_key: str | None = Field(default=None, alias="NVD_API_KEY")

    def effective_embed_dim(self) -> int:
        if self.embed_dim and self.embed_dim > 0:
            return self.embed_dim
        return KNOWN_EMBED_DIMS.get(self.embed_model.lower(), 1024)

    def resolved_vault(self) -> Path:
        p = self.vault_path
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    def resolved_db(self) -> Path:
        p = self.db_path
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
