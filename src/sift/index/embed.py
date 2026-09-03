"""Text embeddings.

Two backends:
  * **fastembed** (ONNX, CPU) — the default. No PyTorch. Fine for incremental
    adds; a full first ingest of thousands of docs is slow (~20 texts/s for
    bge-large). For CPU-only bulk loads use a smaller model
    (``SIFT_EMBED_MODEL=BAAI/bge-base-en-v1.5`` or ``bge-small-en-v1.5``).
  * **sentence-transformers + torch** (GPU) — install with ``uv sync --extra gpu``.
    The torch wheel bundles its own CUDA, so an NVIDIA driver is all you need
    (no system CUDA/cuDNN install). ~50-100x faster than CPU.

Device (``SIFT_EMBED_DEVICE``): ``auto`` (default: GPU if torch+CUDA available,
else CPU), ``cuda``, or ``cpu``.

BGE / Arctic / Nomic / E5 models want an instruction prefix on the *query* side;
we apply the right one based on the model name.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

from sift.config import get_settings

_PREFIXES: dict[str, tuple[str, str]] = {  # (query_prefix, passage_prefix)
    "bge": ("Represent this sentence for searching relevant passages: ", ""),
    "snowflake/snowflake-arctic": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    "nomic-ai/nomic-embed": ("search_query: ", "search_document: "),
    "intfloat/multilingual-e5": ("query: ", "passage: "),
    "intfloat/e5": ("query: ", "passage: "),
}


def _prefixes_for(model_name: str) -> tuple[str, str]:
    low = model_name.lower()
    for key, val in _PREFIXES.items():
        if key in low:
            return val
    return ("", "")


def _torch_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


class Embedder:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.query_prefix, self.passage_prefix = _prefixes_for(model_name)
        if device == "auto":
            device = "cuda" if _torch_cuda() else "cpu"
        elif device == "cuda" and not _torch_cuda():
            print(
                "  ! SIFT_EMBED_DEVICE=cuda but torch+CUDA is unavailable "
                "(run `uv sync --extra gpu`); using CPU"
            )
            device = "cpu"
        self.device = device
        self._backend: tuple[str, object] | None = None  # lazy

    def _load(self):
        if self._backend is not None:
            return self._backend
        if self.device == "cuda":
            from sentence_transformers import SentenceTransformer

            self._backend = ("st", SentenceTransformer(self.model_name, device="cuda"))
        else:
            from fastembed import TextEmbedding

            self._backend = ("fe", TextEmbedding(model_name=self.model_name))
        return self._backend

    def _encode(self, texts: list[str], *, batch_size: int) -> list[list[float]]:
        if not texts:
            return []
        kind, model = self._load()
        if kind == "st":
            arr = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 512,
            )
            return [row.tolist() for row in arr]
        return [vec.tolist() for vec in model.embed(texts, batch_size=batch_size)]

    def embed(
        self,
        texts: Sequence[str] | Iterable[str],
        *,
        kind: str = "passage",
        batch_size: int = 64,
    ) -> list[list[float]]:
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        return self._encode([prefix + t for t in texts], batch_size=batch_size)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], kind="query")[0]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text], kind="passage")[0]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    return Embedder(s.embed_model, s.embed_device)
