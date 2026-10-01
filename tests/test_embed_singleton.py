"""Embedder loading: one model per process, lazy device, optional query backend.

Every model here is a fake installed in ``sys.modules`` in place of fastembed /
sentence-transformers, so nothing downloads and torch is never imported.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types

import pytest


class _Vec(list):
    """Stands in for the numpy rows both backends return."""

    def tolist(self):
        return list(self)


class _Counter:
    def __init__(self) -> None:
        self.n = 0
        self.texts: list[str] = []
        self.lock = threading.Lock()

    def bump(self) -> None:
        with self.lock:
            self.n += 1


def _fake_fastembed(monkeypatch, *, delay: float = 0.0, fail: Exception | None = None):
    """Install a counting fake ``fastembed.TextEmbedding``; returns its counter."""
    counter = _Counter()

    class TextEmbedding:
        def __init__(self, model_name: str, **_kw):
            counter.bump()
            time.sleep(delay)
            if fail is not None:
                raise fail
            self.model_name = model_name

        def embed(self, texts, batch_size=64):
            for t in texts:
                counter.texts.append(t)
                yield _Vec([0.0, 1.0])

    mod = types.ModuleType("fastembed")
    mod.TextEmbedding = TextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    return counter


def _fake_sentence_transformers(monkeypatch):
    counter = _Counter()

    class SentenceTransformer:
        def __init__(self, model_name: str, device: str = "cpu"):
            counter.bump()
            self.device = device

        def encode(self, texts, *, batch_size, normalize_embeddings, show_progress_bar):
            counter.texts.extend(texts)
            return [_Vec([1.0, 0.0]) for _ in texts]

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return counter


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # Keep a SIFT_QUERY_DEVICE from the developer's .env out of these tests.
    monkeypatch.setenv("SIFT_QUERY_DEVICE", "")
    from sift import config
    from sift.index import embed

    config.get_settings.cache_clear()
    embed.get_embedder.cache_clear()
    yield
    embed.get_embedder.cache_clear()


@pytest.fixture
def records():
    """Warnings from sift.index.*, captured on the logger itself (immune to propagate=False)."""
    got: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            got.append(record)

    handler = _Keep(level=logging.WARNING)
    logger = logging.getLogger("sift.index")
    logger.addHandler(handler)
    yield got
    logger.removeHandler(handler)


def test_concurrent_first_calls_load_the_model_once(monkeypatch):
    """FastMCP runs sync tools on a thread pool: N first searches used to load N copies."""
    from sift.index import embed

    loads = _fake_fastembed(monkeypatch, delay=0.2)
    barrier = threading.Barrier(4)
    seen: list[object] = []
    vectors: list[list[float]] = []
    errors: list[BaseException] = []

    def first_search() -> None:
        try:
            barrier.wait(timeout=5)
            emb = embed.get_embedder()
            seen.append(emb)
            vectors.append(emb.embed_query("ssrf via pdf renderer"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=first_search) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert vectors == [[0.0, 1.0]] * 4  # every caller got its vector
    assert len({id(e) for e in seen}) == 1  # one Embedder instance
    assert loads.n == 1  # and one model load

    # Control: the counter does count loads. A different (model, device) is a second one.
    embed.Embedder("some/other-model", "cpu").embed_query("x")
    assert loads.n == 2


def test_warmup_thread_and_first_tool_call_share_one_load(monkeypatch):
    from sift.index import embed

    loads = _fake_fastembed(monkeypatch, delay=0.3)
    result: list[bool] = []
    warm = threading.Thread(target=lambda: result.append(embed.warmup()), daemon=True)
    warm.start()
    time.sleep(0.05)  # the tool call lands while the warm-up is mid-load
    vec = embed.get_embedder().embed_query("jwt alg confusion")
    warm.join(timeout=10)

    assert vec == [0.0, 1.0]
    assert result == [True]
    assert loads.n == 1


def test_failed_warmup_is_logged_never_raised_and_retried_later(monkeypatch, capsys, records):
    from sift.index import embed

    attempts = _fake_fastembed(monkeypatch, fail=RuntimeError("weights not cached"))
    assert embed.warmup() is False
    assert attempts.n == 1
    assert capsys.readouterr().out == ""  # stdout is the MCP JSON-RPC channel
    assert any("warm-up failed" in r.getMessage() for r in records)

    # A failed load caches nothing: once the model is loadable, the next call works.
    loads = _fake_fastembed(monkeypatch)
    assert embed.warmup() is True
    assert loads.n == 1


def test_building_an_embedder_does_not_import_torch(monkeypatch):
    from sift.index import embed

    probes = _Counter()

    def fake_torch_cuda() -> bool:
        probes.bump()
        return False

    monkeypatch.setattr(embed, "_torch_cuda", fake_torch_cuda)
    emb = embed.Embedder("BAAI/bge-small-en-v1.5", "auto")
    assert probes.n == 0  # construction is free...
    assert emb.device == "cpu"  # ...the device resolves on first use
    assert probes.n == 1
    assert emb.device == "cpu"
    assert probes.n == 1  # and only once


def test_cuda_fallback_warning_goes_to_the_log_not_stdout(monkeypatch, capsys, records):
    """The old print() landed in stdout's buffer and was flushed onto the JSON-RPC pipe."""
    from sift.index import embed

    monkeypatch.setattr(embed, "_torch_cuda", lambda: False)
    assert embed.Embedder("m", "cuda").device == "cpu"
    assert embed.Embedder("m2", "cuda").device == "cpu"

    assert capsys.readouterr().out == ""
    hits = [r for r in records if "SIFT_EMBED_DEVICE=cuda" in r.getMessage()]
    assert len(hits) == 1  # once per process, not once per embedder
    assert hits[0].name == "sift.index.embed"  # under the "sift" logger the CLI configures


def test_query_backend_on_cpu_keeps_passages_on_cuda(monkeypatch):
    from sift.index import embed

    monkeypatch.setattr(embed, "_torch_cuda", lambda: True)
    cpu = _fake_fastembed(monkeypatch)
    gpu = _fake_sentence_transformers(monkeypatch)
    emb = embed.Embedder("BAAI/bge-small-en-v1.5", "cuda", query_device="cpu")

    assert emb.embed_query("idor in graphql") == [0.0, 1.0]
    assert (cpu.n, gpu.n) == (1, 0)  # a search-only session never loads the torch model
    assert cpu.texts == [emb.query_prefix + "idor in graphql"]  # bge query prefix kept

    assert emb.embed(["passage text"]) == [[1.0, 0.0]]
    assert (cpu.n, gpu.n) == (1, 1)  # passages still go to the configured device
    assert gpu.texts == ["passage text"]
    assert (emb.device, emb.query_device) == ("cuda", "cpu")


def test_without_a_query_device_queries_share_the_passage_model(monkeypatch):
    from sift.index import embed

    loads = _fake_fastembed(monkeypatch)
    emb = embed.Embedder("m", "cpu")
    emb.embed_query("q")
    emb.embed_one("p")
    assert loads.n == 1
    assert emb.query_device == emb.device == "cpu"


def test_query_backend_failure_falls_back_to_the_passage_model(monkeypatch, capsys, records):
    from sift.index import embed

    monkeypatch.setattr(embed, "_torch_cuda", lambda: True)
    cpu = _fake_fastembed(monkeypatch, fail=OSError("onnx weights not cached, HF_HUB_OFFLINE=1"))
    gpu = _fake_sentence_transformers(monkeypatch)
    emb = embed.Embedder("m", "cuda", query_device="cpu")

    assert emb.embed_query("q1") == [1.0, 0.0]  # served by the passage backend
    assert emb.embed_query("q2") == [1.0, 0.0]
    assert cpu.n == 1  # the broken query backend is not retried on every search
    assert gpu.n == 1
    assert capsys.readouterr().out == ""
    assert sum("query backend" in r.getMessage() for r in records) == 1


def test_get_embedder_reads_the_query_device_setting(monkeypatch):
    from sift.index import embed

    with_setting = types.SimpleNamespace(embed_model="m", embed_device="cpu", query_device="cpu ")
    monkeypatch.setattr(embed, "get_settings", lambda: with_setting)
    assert embed.get_embedder().query_device == "cpu"
    assert embed.get_embedder()._requested_query_device == "cpu"

    # Settings without the field (or with it empty) keep the shared backend.
    embed.get_embedder.cache_clear()
    without = types.SimpleNamespace(embed_model="m", embed_device="cpu")
    monkeypatch.setattr(embed, "get_settings", lambda: without)
    assert embed.get_embedder()._requested_query_device is None


def test_get_embedder_is_a_singleton_that_cache_clear_resets(monkeypatch):
    from sift.index import embed

    loads = _fake_fastembed(monkeypatch)
    first = embed.get_embedder()
    assert embed.get_embedder() is first
    first.embed_query("q")
    assert loads.n == 1

    embed.get_embedder.cache_clear()  # what tests/conftest.py calls between tests
    second = embed.get_embedder()
    assert second is not first
    second.embed_query("q")
    assert loads.n == 2  # the loaded model was forgotten too


def test_backward_compatible_surface(monkeypatch):
    from sift.index import embed

    _fake_fastembed(monkeypatch)
    emb = embed.Embedder("BAAI/bge-small-en-v1.5", "cpu")
    kind, model = emb._load()  # old tuple unpacking
    assert kind == "fe" and model is not None
    assert emb._encode(["a"], batch_size=8) == [[0.0, 1.0]]  # old signature
    assert emb.embed(t for t in ["x", "y"]) == [[0.0, 1.0], [0.0, 1.0]]  # any iterable
    assert emb.embed_one("z") == [0.0, 1.0]
    assert emb.query_prefix.startswith("Represent this sentence")
    assert emb.passage_prefix == ""
