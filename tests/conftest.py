from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point sift at throwaway vault/db dirs and reset the settings cache."""
    vault = tmp_path / "vault"
    db = tmp_path / "db"
    vault.mkdir()
    db.mkdir()
    monkeypatch.setenv("SIFT_VAULT_PATH", str(vault))
    monkeypatch.setenv("SIFT_DB_PATH", str(db))
    monkeypatch.setenv("SIFT_EMBED_DEVICE", "cpu")

    from sift import config

    config.get_settings.cache_clear()
    try:
        from sift.index import embed

        embed.get_embedder.cache_clear()
    except Exception:
        pass
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def vault_path() -> Path:
    from sift.config import get_settings

    return get_settings().resolved_vault()
