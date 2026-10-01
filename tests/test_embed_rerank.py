"""Reranker: a supported default, failures that never break search, sane scores.

fastembed's cross-encoder (and sentence-transformers, where needed) is replaced by
fakes in ``sys.modules``, so nothing downloads.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types

import pytest


class FakeCrossEncoder:
    supported = ["BAAI/bge-reranker-base", "jinaai/jina-reranker-v2-base-multilingual"]
    constructed = 0
    calls: list[tuple[str, list[str]]] = []
    fail_init: Exception | None = None
    fail_rerank: Exception | None = None
    init_delay = 0.0
    lock = threading.Lock()

    def __init__(self, model_name: str, **_kw):
        cls = type(self)
        with cls.lock:
            cls.constructed += 1
        time.sleep(cls.init_delay)
        if cls.fail_init is not None:
            raise cls.fail_init
        self.model_name = model_name

    @classmethod
    def list_supported_models(cls):
        return [{"model": m} for m in cls.supported]

    def rerank(self, query, documents, batch_size=64, **_kw):
        docs = list(documents)
        type(self).calls.append((query, docs))
        if type(self).fail_rerank is not None:
            raise type(self).fail_rerank
        for d in docs:  # the logit is encoded in the excerpt, e.g. "logit=-2.0"
            yield float(d.rsplit("logit=", 1)[1]) if "logit=" in d else 0.0


@pytest.fixture(autouse=True)
def _fake_fastembed(monkeypatch):
    FakeCrossEncoder.constructed = 0
    FakeCrossEncoder.calls = []
    FakeCrossEncoder.fail_init = None
    FakeCrossEncoder.fail_rerank = None
    FakeCrossEncoder.init_delay = 0.0

    root = types.ModuleType("fastembed")
    sub = types.ModuleType("fastembed.rerank")
    leaf = types.ModuleType("fastembed.rerank.cross_encoder")
    leaf.TextCrossEncoder = FakeCrossEncoder
    root.rerank = sub
    sub.cross_encoder = leaf
    monkeypatch.setitem(sys.modules, "fastembed", root)
    monkeypatch.setitem(sys.modules, "fastembed.rerank", sub)
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", leaf)

    monkeypatch.setenv("SIFT_RERANK", "true")
    monkeypatch.setenv("SIFT_RERANK_MODEL", "BAAI/bge-reranker-base")
    monkeypatch.setenv("SIFT_QUALITY_WEIGHT", "1.0")
    monkeypatch.setenv("SIFT_RECENCY_WEIGHT", "1.0")
    from sift import config
    from sift.index import rerank

    config.get_settings.cache_clear()
    rerank.get_reranker.cache_clear()
    yield
    rerank.get_reranker.cache_clear()
    config.get_settings.cache_clear()


@pytest.fixture
def records():
    got: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            got.append(record)

    handler = _Keep(level=logging.WARNING)
    logger = logging.getLogger("sift.index")
    logger.addHandler(handler)
    yield got
    logger.removeHandler(handler)


def _hit(title: str, logit: float | None = None, *, quality: int = 50, score: float = 0.0):
    excerpt = f"excerpt of {title}" + (f" logit={logit}" if logit is not None else "")
    return types.SimpleNamespace(
        title=title,
        heading="Steps > PoC",
        excerpt=excerpt,
        score=score,
        quality=quality,
        created_ts=0.0,
    )


def test_unloadable_model_disables_rerank_once_without_touching_stdout(capsys, records):
    from sift.index import rerank

    FakeCrossEncoder.fail_init = RuntimeError("model weights not cached")
    assert rerank.get_reranker() is None
    assert rerank.get_reranker() is None
    assert FakeCrossEncoder.constructed == 1  # not retried on every search
    assert capsys.readouterr().out == ""  # stdout is the MCP JSON-RPC channel
    assert sum("unavailable" in r.getMessage() for r in records) == 1

    # Control: with a loadable model the same path hands back a working reranker.
    FakeCrossEncoder.fail_init = None
    rerank.get_reranker.cache_clear()
    assert isinstance(rerank.get_reranker(), rerank.Reranker)


def test_rerank_off_loads_nothing(monkeypatch):
    from sift import config
    from sift.index import rerank

    monkeypatch.setenv("SIFT_RERANK", "false")
    config.get_settings.cache_clear()
    assert rerank.get_reranker() is None
    assert FakeCrossEncoder.constructed == 0


def test_model_loads_eagerly_in_get_reranker():
    """The old get_reranker only stored the name, so its fallback could never fire."""
    from sift.index import rerank

    reranker = rerank.get_reranker()
    assert reranker is not None
    assert FakeCrossEncoder.constructed == 1  # before any search has run
    assert rerank.get_reranker() is reranker


def test_inference_error_keeps_the_fused_order(capsys, records):
    from sift.index import rerank

    hits = [_hit("first", 0.0, score=0.3), _hit("second", 5.0, score=0.2), _hit("third", 9.0)]
    FakeCrossEncoder.fail_rerank = RuntimeError("onnxruntime: bad input")
    reranker = rerank.get_reranker()

    out = reranker.rerank("q", list(hits), top_k=2)
    assert [h.title for h in out] == ["first", "second"]
    assert [h.score for h in out] == [0.3, 0.2]  # fused scores untouched
    reranker.rerank("q", list(hits), top_k=2)
    assert len(FakeCrossEncoder.calls) == 1  # disabled after the first failure
    assert capsys.readouterr().out == ""
    assert sum("failed" in r.getMessage() for r in records) == 1

    # Control: a healthy reranker does reorder the same hits.
    FakeCrossEncoder.fail_rerank = None
    healthy = rerank.Reranker("BAAI/bge-reranker-base")
    assert [h.title for h in healthy.rerank("q", list(hits), top_k=2)] == ["third", "second"]


def test_logits_are_squashed_before_the_quality_boost():
    """Multiplying a negative logit by a >1 boost would push the better note down."""
    from sift.index import rerank

    plain = _hit("low quality", -2.0, quality=0)
    good = _hit("high quality", -2.2, quality=100)
    out = rerank.get_reranker().rerank("q", [plain, good], top_k=2)

    # sigmoid(-2.0) * 0.6 = 0.072 < sigmoid(-2.2) * 1.4 = 0.140. Raw logits (the old
    # behaviour) or logit * boost would both have put "low quality" first.
    assert [h.title for h in out] == ["high quality", "low quality"]
    assert all(0.0 < h.score < 1.4 for h in out)

    # Control: at equal quality the cross-encoder's verdict decides.
    a, b = _hit("a", -2.0, quality=50), _hit("b", -2.2, quality=50)
    assert [h.title for h in rerank.get_reranker().rerank("q", [b, a], top_k=2)] == ["a", "b"]


def test_boost_follows_store_multipliers_public_or_private(monkeypatch, capsys, records):
    from sift.index import rerank, store

    hit = _hit("h", quality=100)
    # Control: today's underscore-private multipliers are found (1.4 at quality 100).
    assert rerank._boost(hit) == pytest.approx(1.4)

    # A public rename keeps the boost working.
    monkeypatch.setattr(store, "quality_mult", lambda q: 2.0, raising=False)
    monkeypatch.setattr(store, "recency_mult", lambda ts: 1.5, raising=False)
    assert rerank._boost(hit) == pytest.approx(3.0)

    # Gone entirely: neutral boost, never an exception, and one warning, not one per hit.
    for name in ("quality_mult", "_quality_mult"):
        monkeypatch.delattr(store, name)
    assert rerank._boost(hit) == 1.0
    assert rerank._boost(hit) == 1.0
    assert sum("boost unavailable" in r.getMessage() for r in records) == 1
    assert capsys.readouterr().out == ""


def test_sigmoid_is_overflow_safe():
    from sift.index.rerank import _sigmoid

    assert _sigmoid(0.0) == 0.5
    assert _sigmoid(1000.0) == 1.0
    assert _sigmoid(-1000.0) == 0.0
    assert _sigmoid(float("nan")) == 0.0
    assert _sigmoid(-3.0) < _sigmoid(-2.0) < _sigmoid(2.0)


def test_cross_encoder_sees_title_heading_and_excerpt():
    from sift.index import rerank

    rerank.get_reranker().rerank(
        "q", [_hit("Stored XSS via SVG", 1.0), _hit("Other", 0.0)], top_k=2
    )
    _query, docs = FakeCrossEncoder.calls[0]
    assert docs[0] == "Stored XSS via SVG\nSteps > PoC\nexcerpt of Stored XSS via SVG logit=1.0"

    bare = types.SimpleNamespace(
        title="Title only", heading="", excerpt="", score=0.0, quality=50, created_ts=0.0
    )
    assert rerank._doc_text(bare) == "Title only"


def test_unsupported_model_without_sentence_transformers_is_reported(monkeypatch, records):
    from sift import config
    from sift.index import rerank

    monkeypatch.setenv("SIFT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # gpu extra absent
    config.get_settings.cache_clear()

    assert rerank.get_reranker() is None
    assert FakeCrossEncoder.constructed == 0  # never pushed an unsupported name at fastembed
    message = " ".join(r.getMessage() for r in records)
    assert "bge-reranker-v2-m3" in message and rerank.DEFAULT_RERANK_MODEL in message


def test_unsupported_model_loads_through_sentence_transformers(monkeypatch):
    from sift import config
    from sift.index import rerank

    seen: dict[str, object] = {}

    class CrossEncoder:
        def __init__(self, model_name: str, device: str = "cpu"):
            seen["model"], seen["device"] = model_name, device

        def predict(self, pairs, *, show_progress_bar, activation_fn):
            seen["activation_fn"] = activation_fn
            return [float(d.rsplit("logit=", 1)[1]) for _q, d in pairs]

    st = types.ModuleType("sentence_transformers")
    st.CrossEncoder = CrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setenv("SIFT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    config.get_settings.cache_clear()

    reranker = rerank.get_reranker()
    assert reranker is not None
    assert seen["model"] == "BAAI/bge-reranker-v2-m3"
    assert seen["device"] == "cpu"  # conftest pins SIFT_EMBED_DEVICE=cpu
    out = reranker.rerank("q", [_hit("low", -1.0), _hit("high", 3.0)], top_k=2)
    assert [h.title for h in out] == ["high", "low"]
    assert seen["activation_fn"](3.0) == 3.0  # raw logits, squashed by us exactly once


def test_old_unsupported_default_is_only_used_when_chosen():
    from sift.index import rerank

    def settings(name, explicit):
        return types.SimpleNamespace(
            rerank_model=name, model_fields_set={"rerank_model"} if explicit else set()
        )

    legacy = "BAAI/bge-reranker-v2-m3"
    assert rerank._configured_model(settings(legacy, False)) == rerank.DEFAULT_RERANK_MODEL
    assert rerank._configured_model(settings(legacy, True)) == legacy
    assert rerank._configured_model(settings("", True)) == rerank.DEFAULT_RERANK_MODEL
    other = "jinaai/jina-reranker-v2-base-multilingual"
    assert rerank._configured_model(settings(other, False)) == other


def test_concurrent_first_searches_load_the_reranker_once():
    from sift.index import rerank

    FakeCrossEncoder.init_delay = 0.2
    barrier = threading.Barrier(4)
    got: list[object] = []

    def first_search() -> None:
        barrier.wait(timeout=5)
        got.append(rerank.get_reranker())

    threads = [threading.Thread(target=first_search) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(got) == 4 and got[0] is not None
    assert len({id(r) for r in got}) == 1
    assert FakeCrossEncoder.constructed == 1


def test_warmup_reports_and_never_raises(monkeypatch):
    from sift import config
    from sift.index import rerank

    assert rerank.warmup() is True

    rerank.get_reranker.cache_clear()
    FakeCrossEncoder.fail_init = RuntimeError("offline")
    assert rerank.warmup() is False

    monkeypatch.setenv("SIFT_RERANK", "false")
    config.get_settings.cache_clear()
    rerank.get_reranker.cache_clear()
    assert rerank.warmup() is False
