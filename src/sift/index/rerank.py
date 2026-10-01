"""Optional cross-encoder reranking.

Off by default (``SIFT_RERANK=false``). When on, it re-scores the fused top-N against
the query — a precision gain for a model download and ~100-300 ms per query on CPU.

Model (``SIFT_RERANK_MODEL``, default ``BAAI/bge-reranker-base``). fastembed's ONNX
cross-encoder serves ``BAAI/bge-reranker-base``,
``jinaai/jina-reranker-v2-base-multilingual``, ``jinaai/jina-reranker-v1-turbo-en``,
``jinaai/jina-reranker-v1-tiny-en`` and ``Xenova/ms-marco-MiniLM-L-6-v2`` /
``-L-12-v2``. Any other model, ``BAAI/bge-reranker-v2-m3`` included, needs
sentence-transformers (``uv sync --extra gpu``) and loads through its CrossEncoder.

Reranking is an optional precision upgrade, so it never breaks search: the model
loads eagerly in :func:`get_reranker`, which logs one warning and returns None when it
cannot load, and an inference error disables the reranker for the process and keeps
the fused order. Diagnostics go through ``logging`` (stderr), never stdout, which is
the MCP server's JSON-RPC channel.

Scoring: the cross-encoder judges title, heading and excerpt together. Its raw logit
(often negative) is squashed to (0, 1) with a sigmoid and then multiplied by the
quality/recency multipliers the fused score gets. Multiplying the raw logit instead
would push a better note *down*.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from sift.config import get_settings

if TYPE_CHECKING:
    from sift.index.store import Hit

log = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
# The old built-in default, which fastembed cannot load. Only honoured when set explicitly.
_LEGACY_DEFAULT = "BAAI/bge-reranker-v2-m3"


def _sigmoid(x: float) -> float:
    """Logistic squash that cannot overflow; NaN counts as irrelevant."""
    if math.isnan(x):
        return 0.0
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _doc_text(h: Hit) -> str:
    """What the cross-encoder judges: the note's identity plus the matched excerpt."""
    return "\n".join(p for p in (h.title, h.heading, h.excerpt) if p)


_BOOST_WARNED = threading.Event()


def _store_mult(store, name: str):
    """store's ``quality_mult`` / ``recency_mult``, public or underscore-private."""
    return getattr(store, name, None) or getattr(store, f"_{name}", None)


def _boost(h: Hit) -> float:
    """The fused score's quality x recency multiplier, so SIFT_*_WEIGHT keeps working.

    Never fails a search: if the multipliers are gone or raise, the boost is 1.0 and one
    warning says so, because a silently dropped boost is a silent ranking change.
    """
    try:
        from sift.index import store

        quality_mult = _store_mult(store, "quality_mult")
        recency_mult = _store_mult(store, "recency_mult")
        if quality_mult is None or recency_mult is None:
            raise AttributeError("sift.index.store has no quality/recency multiplier")
        return float(quality_mult(getattr(h, "quality", 0))) * float(
            recency_mult(getattr(h, "created_ts", 0.0))
        )
    except Exception as exc:  # noqa: BLE001 - a missing boost must not fail the search
        if not _BOOST_WARNED.is_set():
            _BOOST_WARNED.set()
            log.warning(
                "rerank: quality/recency boost unavailable (%s: %s); ranking by relevance only",
                type(exc).__name__,
                exc,
            )
        return 1.0


# ---- cross-encoder backends: (query, docs) -> raw relevance logits -----------


def _identity(x):
    return x


def _fastembed_supports(cls, model_name: str) -> bool:
    try:
        names = {str(m.get("model", "")).lower() for m in cls.list_supported_models()}
    except Exception:  # noqa: BLE001 - cannot tell: let the constructor decide
        return True
    return model_name.lower() in names


def _load_cross_encoder(model_name: str) -> Callable[[str, list[str]], list[float]]:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    if _fastembed_supports(TextCrossEncoder, model_name):
        encoder = TextCrossEncoder(model_name=model_name)

        def score_fastembed(query: str, docs: list[str]) -> list[float]:
            return [float(s) for s in encoder.rerank(query, docs)]

        return score_fastembed

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ValueError(
            f"{model_name} is not supported by fastembed's TextCrossEncoder (try "
            f"{DEFAULT_RERANK_MODEL}); other models need sentence-transformers "
            "(`uv sync --extra gpu`)"
        ) from None
    from sift.index.embed import _resolve_device

    device = _resolve_device(get_settings().embed_device)
    model = CrossEncoder(model_name, device=device)

    def score_st(query: str, docs: list[str]) -> list[float]:
        # Raw logits (no activation), the same scale fastembed returns.
        arr = model.predict(
            [(query, d) for d in docs], show_progress_bar=False, activation_fn=_identity
        )
        return [float(s) for s in arr]

    return score_st


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: Callable[[str, list[str]], list[float]] | None = None  # lazy
        self._lock = threading.Lock()  # single-flight load; serialised inference
        self._broken = False

    def _load(self) -> Callable[[str, list[str]], list[float]]:
        model = self._model
        if model is None:
            with self._lock:
                if self._model is None:
                    self._model = _load_cross_encoder(self.model_name)
                model = self._model
        return model

    def rerank(self, query: str, hits: list[Hit], *, top_k: int) -> list[Hit]:
        if self._broken or len(hits) <= 1:
            return hits[:top_k]
        docs = [_doc_text(h) for h in hits]
        try:
            score = self._load()
            with self._lock:
                scores = list(score(query, docs))
            if len(scores) != len(hits):
                raise ValueError(f"{len(scores)} scores for {len(hits)} passages")
        except Exception as exc:  # noqa: BLE001 - reranking is optional; never fail a search
            self._broken = True
            log.warning(
                "reranker %s failed (%s: %s); disabled for this process, keeping the fused order",
                self.model_name,
                type(exc).__name__,
                exc,
            )
            return hits[:top_k]
        for h, s in zip(hits, scores, strict=True):
            h.score = _sigmoid(float(s)) * _boost(h)
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]


def _configured_model(s) -> str:
    """SIFT_RERANK_MODEL, except that the unsupported old default is only used if set."""
    name = str(getattr(s, "rerank_model", "") or "").strip()
    explicit = "rerank_model" in (getattr(s, "model_fields_set", None) or ())
    if not name or (name == _LEGACY_DEFAULT and not explicit):
        return DEFAULT_RERANK_MODEL
    return name


_BUILD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _build_reranker(enabled: bool, model_name: str) -> Reranker | None:
    if not enabled:
        return None
    reranker = Reranker(model_name)
    try:
        reranker._load()  # eager: a missing or unsupported model surfaces here, once
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "reranker %s unavailable (%s: %s); continuing without it",
            model_name,
            type(exc).__name__,
            exc,
        )
        return None
    return reranker


def get_reranker() -> Reranker | None:
    """The process-wide reranker, or None when SIFT_RERANK is off or the model won't load."""
    s = get_settings()
    with _BUILD_LOCK:  # one load even when the first searches arrive together
        return _build_reranker(bool(s.rerank), _configured_model(s))


def _reset() -> None:
    """Forget the cached reranker and the one-time boost warning (tests call this)."""
    with _BUILD_LOCK:
        _build_reranker.cache_clear()
    _BOOST_WARNED.clear()


get_reranker.cache_clear = _reset  # type: ignore[attr-defined]


def warmup() -> bool:
    """Load the reranker now if SIFT_RERANK is on. Never raises; True when one is ready."""
    try:
        return get_reranker() is not None
    except Exception as exc:  # noqa: BLE001
        log.warning("reranker warm-up failed (%s: %s)", type(exc).__name__, exc)
        return False
