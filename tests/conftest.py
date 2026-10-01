from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

# Credentials the suite must never see: a live key in the developer's environment
# would turn an offline test into a paid or rate-limited network call. The opt-in
# `calibration` tests keep them (they are the live gate check).
_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_PROFILE",
    # Deleted, never set: the SDK treats a set ANTHROPIC_CONFIG_DIR as an explicit
    # profile choice and raises instead of reporting "no credentials".
    "ANTHROPIC_CONFIG_DIR",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "H1_API_USERNAME",
    "H1_API_TOKEN",
    "NVD_API_KEY",
    "HF_TOKEN",
)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Point sift at throwaway vault/db dirs, away from the developer's real config.

    - The project `.env` is not read: it holds live credentials, a real Chromium
      checkout path and possibly a non-default model, and CI has none of them.
    - Every inherited SIFT_* variable is dropped, so only what a test sets applies.
    - Credentials are removed (except for `calibration` tests), and the SDK's
      on-disk profile dir is moved to an empty temp dir.
    """
    from sift import config

    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    for key in list(os.environ):
        if key.startswith("SIFT_"):
            monkeypatch.delenv(key)
    if request.node.get_closest_marker("calibration") is None:
        for key in _CREDENTIAL_VARS:
            monkeypatch.delenv(key, raising=False)
        # The Anthropic SDK's platform default profile dir (`ant auth login`).
        if sys.platform == "win32":
            monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        else:
            monkeypatch.setenv("HOME", str(tmp_path / "home"))

    vault = tmp_path / "vault"
    db = tmp_path / "db"
    vault.mkdir()
    db.mkdir()
    monkeypatch.setenv("SIFT_VAULT_PATH", str(vault))
    monkeypatch.setenv("SIFT_DB_PATH", str(db))
    monkeypatch.setenv("SIFT_EMBED_DEVICE", "cpu")

    config.get_settings.cache_clear()
    try:
        from sift.index import embed, rerank

        embed.get_embedder.cache_clear()
        rerank.get_reranker.cache_clear()
    except Exception:
        pass
    # Module-level state of modules a test already imported (importing them here
    # would pull LanceDB into every test).
    pipeline = sys.modules.get("sift.pipeline")
    if pipeline is not None and hasattr(pipeline, "_SYNC_STATES"):
        monkeypatch.setattr(pipeline, "_SYNC_STATES", {})
    server = sys.modules.get("sift.mcp_server")
    if server is not None:
        for flag in ("_background_enabled", "_background_started"):
            if hasattr(server, flag):
                monkeypatch.setattr(server, flag, False)
    # The CLI configures the `sift` logger (stderr handler, level, no propagation). Put
    # it back, or every later test's caplog would see a different logger.
    sift_logger = logging.getLogger("sift")
    saved = (list(sift_logger.handlers), sift_logger.level, sift_logger.propagate)
    yield
    sift_logger.handlers[:] = saved[0]
    sift_logger.setLevel(saved[1])
    sift_logger.propagate = saved[2]
    config.get_settings.cache_clear()


@pytest.fixture
def vault_path() -> Path:
    from sift.config import get_settings

    return get_settings().resolved_vault()


FAKE_DIM = 64


class FakeEmbedder:
    """A deterministic bag-of-words embedder: no model, no network. Counts passages."""

    model_name = "fake"
    device = "cpu"
    query_prefix = ""
    passage_prefix = ""

    def __init__(self) -> None:
        self.embedded = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * FAKE_DIM
        for i, tok in enumerate(text.lower().split()):
            v[sum(map(ord, tok)) % FAKE_DIM] += 1.0
            if i > 200:
                break
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed(self, texts, *, kind="passage", batch_size=32):
        self.embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def embed_one(self, text):
        return self._vec(text)


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    """Opt-in: replace the embedder process-wide and size the index for it."""
    monkeypatch.setenv("SIFT_EMBED_DIM", str(FAKE_DIM))
    from sift import config

    config.get_settings.cache_clear()
    embedder = FakeEmbedder()
    monkeypatch.setattr("sift.index.embed.get_embedder", lambda: embedder)
    return embedder
