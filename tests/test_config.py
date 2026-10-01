"""Settings: blank values, derived dimensions, new switches, and .env.example coverage."""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

import pytest


def _settings(**kwargs):
    from sift import config

    config.get_settings.cache_clear()
    return config.Settings(**kwargs)


# --------------------------------------------------------------------------- #
# blank values mean "unset"
# --------------------------------------------------------------------------- #
def test_blank_env_file_values_fall_back_to_defaults(tmp_path):
    env = tmp_path / "blank.env"
    env.write_text(
        "SIFT_CHROMIUM_SRC=\nSIFT_EMBED_DIM=\nSIFT_MODEL_CUTOFF=\nSIFT_MCP_WARMUP=\n"
        "SIFT_QUERY_DEVICE=\nANTHROPIC_API_KEY=\n",
        encoding="utf-8",
    )
    s = _settings(_env_file=env)
    assert s.chromium_src is None  # was Path('.'), which looked "set"
    assert s.embed_dim == 0  # was a ValidationError that killed every command
    assert s.model_cutoff == date(2026, 4, 1)
    assert s.mcp_warmup is True
    assert s.query_device == ""
    assert s.anthropic_api_key is None


def test_the_env_file_is_actually_read(tmp_path):
    """Positive control for the test above: a non-blank value in the same file wins."""
    env = tmp_path / "set.env"
    env.write_text("SIFT_EMBED_DIM=384\nSIFT_CHROMIUM_SRC=/src/chromium\n", encoding="utf-8")
    s = _settings(_env_file=env)
    assert s.embed_dim == 384
    assert s.chromium_src == Path("/src/chromium")


def test_blank_env_vars_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("SIFT_CHROMIUM_SRC", "")
    monkeypatch.setenv("SIFT_EMBED_DIM", "")
    s = _settings()
    assert s.chromium_src is None
    assert s.embed_dim == 0

    monkeypatch.setenv("SIFT_EMBED_DIM", "12")  # control: a real value applies
    assert _settings().embed_dim == 12


def test_blank_chromium_src_is_reported_as_not_set(monkeypatch):
    monkeypatch.setenv("SIFT_CHROMIUM_SRC", "")
    from sift import config
    from sift.ingest import chromium_docs

    config.get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="is not set"):
        chromium_docs._resolved_src()


def test_home_relative_paths_are_expanded(monkeypatch):
    monkeypatch.setenv("SIFT_VAULT_PATH", "~/sift-test-vault")
    monkeypatch.setenv("SIFT_CHROMIUM_SRC", "~/chromium/src")
    s = _settings()
    assert s.resolved_vault() == Path("~/sift-test-vault").expanduser()
    assert s.resolved_vault().is_absolute()
    assert "~" not in str(s.chromium_src)


def test_a_gate_effort_typo_does_not_break_settings(monkeypatch):
    """The MCP server loads settings in every tool: a gate-only typo must not stop it.
    The gate validates the value where it is used."""
    monkeypatch.setenv("SIFT_GATE_EFFORT", "extreme")
    assert _settings().gate_effort == "extreme"


# --------------------------------------------------------------------------- #
# embedding dimension: derived, never guessed
# --------------------------------------------------------------------------- #
def test_known_model_dims_come_from_the_table(monkeypatch):
    monkeypatch.setenv("SIFT_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    assert _settings().effective_embed_dim() == 1024


def test_unlisted_model_dim_comes_from_fastembed(monkeypatch):
    monkeypatch.setenv("SIFT_EMBED_MODEL", "Sentence-Transformers/All-MiniLM-L6-v2")  # any case
    assert _settings().effective_embed_dim() == 384


def test_an_unknown_model_raises_instead_of_guessing_1024(monkeypatch):
    monkeypatch.setenv("SIFT_EMBED_MODEL", "someone/unknown-embedder")
    with pytest.raises(ValueError, match="SIFT_EMBED_DIM"):
        _settings().effective_embed_dim()


def test_explicit_embed_dim_wins(monkeypatch):
    monkeypatch.setenv("SIFT_EMBED_MODEL", "someone/unknown-embedder")
    monkeypatch.setenv("SIFT_EMBED_DIM", "256")
    assert _settings().effective_embed_dim() == 256
    monkeypatch.setenv("SIFT_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    assert _settings().effective_embed_dim() == 256


# --------------------------------------------------------------------------- #
# reranker default and the new switches
# --------------------------------------------------------------------------- #
def test_default_reranker_is_one_fastembed_supports():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    from sift.index import rerank

    s = _settings()
    supported = {m["model"] for m in TextCrossEncoder.list_supported_models()}
    assert s.rerank_model == "BAAI/bge-reranker-base"
    assert s.rerank_model in supported
    assert rerank._configured_model(s) == s.rerank_model


def test_new_switch_defaults():
    s = _settings()
    assert s.mcp_warmup is True
    assert s.mcp_auto_sync is True
    assert s.query_device == ""
    assert s.vault_ignore_dirs == ""


def test_new_switches_parse_from_the_environment(monkeypatch):
    monkeypatch.setenv("SIFT_MCP_WARMUP", "false")
    monkeypatch.setenv("SIFT_MCP_AUTO_SYNC", "0")
    monkeypatch.setenv("SIFT_QUERY_DEVICE", "cpu")
    monkeypatch.setenv("SIFT_VAULT_IGNORE_DIRS", "Archive, finding/private")
    s = _settings()
    assert s.mcp_warmup is False
    assert s.mcp_auto_sync is False
    assert s.query_device == "cpu"
    assert s.vault_ignore_dirs == "Archive, finding/private"


def test_mcp_server_switches_read_the_declared_settings(monkeypatch):
    monkeypatch.setenv("SIFT_MCP_WARMUP", "no")
    from sift import config, mcp_server

    config.get_settings.cache_clear()
    assert mcp_server._warmup_on() is False
    assert mcp_server._auto_sync_on() is True


# --------------------------------------------------------------------------- #
# .env.example is the one list of settings
# --------------------------------------------------------------------------- #
def _example_keys() -> tuple[str, set[str]]:
    from sift.config import PROJECT_ROOT

    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    return text, set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, re.M))


def test_env_example_lists_every_setting():
    from sift.config import Settings

    _text, keys = _example_keys()
    aliases = {f.alias for f in Settings.model_fields.values() if f.alias}
    assert aliases, "positive control: Settings declares aliases"
    assert aliases - keys == set()


def test_env_example_has_no_stale_sift_keys():
    from sift.config import Settings

    _text, keys = _example_keys()
    aliases = {f.alias for f in Settings.model_fields.values() if f.alias}
    assert {k for k in keys if k.startswith("SIFT_")} - aliases == set()


def test_env_example_loads_as_is():
    """`sift init` copies .env.example verbatim, so it must load and mean the defaults."""
    from sift.config import PROJECT_ROOT, Settings

    example = _settings(_env_file=PROJECT_ROOT / ".env.example")
    defaults = Settings()
    for field in (
        "embed_model",
        "embed_dim",
        "query_device",
        "rerank",
        "rerank_model",
        "quality_weight",
        "recency_weight",
        "mcp_warmup",
        "mcp_auto_sync",
        "chromium_src",
        "model_cutoff",
        "gate_model",
        "gate_effort",
        "research_feeds",
        "vault_ignore_dirs",
    ):
        assert getattr(example, field) == getattr(defaults, field), field


# --------------------------------------------------------------------------- #
# the suite is isolated from the developer's own config
# --------------------------------------------------------------------------- #
def test_tests_never_read_the_real_env_file():
    from sift import config

    assert config.Settings.model_config["env_file"] is None
    assert {k for k in os.environ if k.startswith("SIFT_")} == {
        "SIFT_VAULT_PATH",
        "SIFT_DB_PATH",
        "SIFT_EMBED_DEVICE",
    }
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "H1_API_TOKEN", "HF_TOKEN"):
        assert key not in os.environ
