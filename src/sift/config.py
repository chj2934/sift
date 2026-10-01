"""Runtime configuration, loaded from environment / .env file."""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
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


@lru_cache(maxsize=8)
def _fastembed_dim(model: str) -> int | None:
    """Output dim of `model` (lower-cased) from fastembed's model list, or None.

    The list is static and offline. Imported lazily: only a model outside
    KNOWN_EMBED_DIMS pays for the fastembed import, once per process.
    """
    try:
        from fastembed import TextEmbedding

        models = TextEmbedding.list_supported_models()
    except Exception:  # noqa: BLE001 - not installed or broken: the caller asks for SIFT_EMBED_DIM
        return None
    for m in models:
        if str(m.get("model", "")).lower() == model:
            try:
                return int(m["dim"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="",
        extra="ignore",
        # A blank `KEY=` (how .env.example ships its optional keys, and `sift init`
        # copies it verbatim) means "unset": the default applies. Without this a blank
        # SIFT_EMBED_DIM= crashed every command, `sift mcp` included, and a blank
        # SIFT_CHROMIUM_SRC= became Path('.'), which defeated the "is not set" check.
        env_ignore_empty=True,
    )

    # --- paths ---
    vault_path: Path = Field(default=PROJECT_ROOT / "vault", alias="SIFT_VAULT_PATH")
    db_path: Path = Field(default=PROJECT_ROOT / "data" / "lancedb", alias="SIFT_DB_PATH")
    # Comma-separated folders every vault walker skips (listings, the catalog,
    # reindex). A bare name matches at any depth; a value with '/' is vault-relative.
    vault_ignore_dirs: str = Field(default="", alias="SIFT_VAULT_IGNORE_DIRS")

    # --- embeddings ---
    embed_model: str = Field(default="BAAI/bge-base-en-v1.5", alias="SIFT_EMBED_MODEL")
    embed_device: str = Field(default="auto", alias="SIFT_EMBED_DEVICE")  # auto|cuda|cpu
    embed_dim: int = Field(default=0, alias="SIFT_EMBED_DIM")  # 0 = derive from model
    # Device for query embeddings only ('' = same as SIFT_EMBED_DEVICE). 'cpu' serves
    # queries from fastembed's ONNX copy of the same model: same rankings, a faster
    # cold start and no VRAM, while passages (indexing) keep SIFT_EMBED_DEVICE.
    query_device: str = Field(default="", alias="SIFT_QUERY_DEVICE")

    # --- retrieval ---
    rerank: bool = Field(default=False, alias="SIFT_RERANK")
    # fastembed 0.8 does not support bge-reranker-v2-m3 (the old default); that one
    # loads only when named explicitly, through sentence-transformers (gpu extra).
    rerank_model: str = Field(default="BAAI/bge-reranker-base", alias="SIFT_RERANK_MODEL")
    # Ranking re-weights. 1.0 = full effect, 0.0 = disable that factor.
    quality_weight: float = Field(default=1.0, alias="SIFT_QUALITY_WEIGHT")
    recency_weight: float = Field(default=1.0, alias="SIFT_RECENCY_WEIGHT")

    # --- MCP server ---
    # Warm the query model, reranker and link graph on a daemon thread once the client
    # has initialised, so the first search skips the cold start. On cuda (and with
    # SIFT_QUERY_DEVICE unset) every concurrent Claude Code session then holds ~1.3 GB
    # of VRAM; SIFT_QUERY_DEVICE=cpu makes the warm-up VRAM-free.
    mcp_warmup: bool = Field(default=True, alias="SIFT_MCP_WARMUP")
    # Sync vault edits (Obsidian) into the index in the background of the MCP server.
    # Changed notes are embedded in the server process; more than 200 changes are
    # left for `sift reindex`.
    mcp_auto_sync: bool = Field(default=True, alias="SIFT_MCP_AUTO_SYNC")

    # --- ingestion ---
    # Extra RSS/Atom feed URLs for `ingest research`, comma-separated.
    research_feeds: str = Field(default="", alias="SIFT_RESEARCH_FEEDS")
    # Local Chromium checkout (the `src` directory). Both Chromium sources read the
    # tree at whatever revision it is synced to and record that SHA on every note,
    # so a note is always traceable to the code it described.
    chromium_src: Path | None = Field(default=None, alias="SIFT_CHROMIUM_SRC")
    # The reasoning model's training cutoff. This is the default horizon for every
    # freshness source: material older than this is, by construction, something the
    # model already knows, and storing it only dilutes ranking (see the novelty gate).
    # Deliberately a little *before* the true cutoff - knowledge thins out near the
    # boundary rather than stopping dead, so the last few weeks are worth re-reading.
    model_cutoff: date = Field(default=date(2026, 4, 1), alias="SIFT_MODEL_CUTOFF")

    # --- novelty gate ---
    # Only the reasoning model can judge what the reasoning model already knows, so
    # this must stay a Claude model — a local model would be guessing.
    gate_model: str = Field(default="claude-opus-5", alias="SIFT_GATE_MODEL")
    # low|medium|high|xhigh|max. Deliberately a plain str: the gate validates it where
    # it is used, so a typo here cannot take down the MCP server or unrelated commands.
    gate_effort: str = Field(default="low", alias="SIFT_GATE_EFFORT")

    # --- external APIs ---
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    h1_api_username: str | None = Field(default=None, alias="H1_API_USERNAME")
    h1_api_token: str | None = Field(default=None, alias="H1_API_TOKEN")
    nvd_api_key: str | None = Field(default=None, alias="NVD_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    @field_validator("vault_path", "db_path", "chromium_src", mode="after")
    @classmethod
    def _expand_user(cls, value: Path | None) -> Path | None:
        # SIFT_VAULT_PATH=~/vault used to resolve to <repo>/~/vault.
        return value.expanduser() if value is not None else None

    def effective_embed_dim(self) -> int:
        """The index's vector width: SIFT_EMBED_DIM when set, else the model's known
        output dim (sift's table, then fastembed's model list).

        Raises ValueError for a model neither knows. Guessing 1024 used to build a
        table the first flush could not write to, after `reindex --force` had already
        dropped the old one.
        """
        if self.embed_dim and self.embed_dim > 0:
            return self.embed_dim
        key = self.embed_model.strip().lower()
        dim = KNOWN_EMBED_DIMS.get(key) or _fastembed_dim(key)
        if dim:
            return dim
        raise ValueError(
            f"unknown output dimension for SIFT_EMBED_MODEL={self.embed_model!r}; "
            "set SIFT_EMBED_DIM to the model's vector width"
        )

    def resolved_vault(self) -> Path:
        p = self.vault_path
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    def resolved_db(self) -> Path:
        p = self.db_path
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    # huggingface_hub / fastembed read HF_TOKEN from the process env, not our .env.
    if s.hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = s.hf_token
    return s
