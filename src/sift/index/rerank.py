"""Optional cross-encoder reranking (bge-reranker-v2-m3).

Off by default (``SIFT_RERANK=false``). When on, it reorders the fused top-N by
query/passage relevance — a meaningful precision gain at the cost of a model
download and ~100-300ms per query on CPU.
"""

from __future__ import annotations

from functools import lru_cache

from sift.config import get_settings
from sift.index.store import Hit


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, hits: list[Hit], *, top_k: int) -> list[Hit]:
        if len(hits) <= 1:
            return hits[:top_k]
        model = self._load()
        scores = list(model.rerank(query, [h.excerpt or h.title for h in hits]))
        for h, s in zip(hits, scores, strict=False):
            h.score = float(s)
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]


@lru_cache
def get_reranker() -> Reranker | None:
    s = get_settings()
    if not s.rerank:
        return None
    try:
        return Reranker(s.rerank_model)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! reranker unavailable ({exc}); continuing without it")
        return None
