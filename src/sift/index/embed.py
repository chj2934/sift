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
else CPU), ``cuda``, or ``cpu``. It is resolved on first use, not when an
:class:`Embedder` is built, so building one never imports torch.

Query backend (``SIFT_QUERY_DEVICE``, optional, default: same as passages). Set it
to ``cpu`` to embed *queries* with fastembed's ONNX export of the same model while
passages (ingest, reindex, ``remember``) stay on ``SIFT_EMBED_DEVICE``. Measured
for bge-large against the CUDA model: query cosine 1.00000, identical top-40, a
~2.9 s cold start instead of ~11 s and no VRAM, so a search-only MCP session never
imports torch. If the query backend cannot load (its ONNX weights are not cached
and ``HF_HUB_OFFLINE=1``, say) queries fall back to the passage backend, with one
warning. fastembed ships *quantized* exports of bge-base/bge-small, whose vectors
are close to, not identical with, the torch model's. ``HF_HUB_OFFLINE=1`` skips the
Hub revision checks at load once every model is cached; changing
``SIFT_EMBED_MODEL`` then needs one online run.

Concurrency: each (model, device) backend loads at most once per process, under a
lock, so concurrent first calls (FastMCP runs sync tools on a thread pool) and a
background :func:`warmup` share one load instead of each putting a copy on the GPU.

Token budget: :meth:`Embedder.count_tokens`, :attr:`Embedder.max_seq_length` and
:meth:`Embedder.passage_budget` read only the model's ``tokenizer.json`` and config
files, never its weights, so chunking can size passages to the real window
(bge-*-v1.5: 512 tokens including [CLS]/[SEP]). Both backends silently drop
whatever lies past it.

BGE / Arctic / Nomic / E5 models want an instruction prefix on the *query* side;
we apply the right one based on the model name.

Diagnostics go through ``logging`` (stderr by default), never ``print``: this
module runs inside the MCP server, whose stdout is the JSON-RPC channel.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from sift.config import get_settings

log = logging.getLogger(__name__)

# Fallback model window (tokens, special tokens included) when no config says otherwise.
DEFAULT_MAX_SEQ_LENGTH = 512
_KNOWN_MAX_SEQ: dict[str, int] = {
    "baai/bge-large-en-v1.5": 512,
    "baai/bge-base-en-v1.5": 512,
    "baai/bge-small-en-v1.5": 512,
    "snowflake/snowflake-arctic-embed-l": 512,
    "snowflake/snowflake-arctic-embed-m": 512,
    "nomic-ai/nomic-embed-text-v1.5": 8192,
    "intfloat/multilingual-e5-large": 512,
}
# tokenizer_config.json writes ~1e30 to mean "no limit"; anything this big is not a window.
_MAX_SANE_WINDOW = 1_000_000

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


# ---- one-time warnings ----------------------------------------------------

_WARNED: set[str] = set()
_WARNED_LOCK = threading.Lock()


def _warn_once(key: str, msg: str, *args: object) -> None:
    with _WARNED_LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    log.warning(msg, *args)


def _resolve_device(requested: str | None, *, setting: str = "SIFT_EMBED_DEVICE") -> str:
    """``auto`` -> cuda if torch+CUDA is available, else cpu; ``cuda`` without CUDA -> cpu."""
    device = (requested or "auto").strip().lower()
    if device == "auto":
        return "cuda" if _torch_cuda() else "cpu"
    if device == "cuda" and not _torch_cuda():
        _warn_once(
            f"no-cuda:{setting}",
            "%s=cuda but torch+CUDA is unavailable (run `uv sync --extra gpu`); using CPU",
            setting,
        )
        return "cpu"
    return device


# ---- model backends: one load per (model, device) per process --------------


@dataclass(eq=False)
class _Backend:
    """A loaded model, plus the lock that serialises inference on it."""

    kind: str  # "st" (sentence-transformers) | "fe" (fastembed)
    model: object
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __iter__(self) -> Iterator[object]:
        # Keeps the old `kind, model = embedder._load()` unpacking working.
        return iter((self.kind, self.model))


_BACKENDS: dict[tuple[str, str], _Backend] = {}
_LOAD_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def _construct_backend(model_name: str, device: str) -> _Backend:
    if device == "cuda":
        from sentence_transformers import SentenceTransformer

        return _Backend("st", SentenceTransformer(model_name, device="cuda"))
    from fastembed import TextEmbedding

    return _Backend("fe", TextEmbedding(model_name=model_name))


def _load_backend(model_name: str, device: str) -> _Backend:
    """The process-wide backend for (model, device), loaded at most once.

    Racing first callers block on the same per-key lock and then reuse the model the
    winner loaded. A failed load caches nothing, so the next call retries it.
    """
    key = (model_name, device)
    backend = _BACKENDS.get(key)
    if backend is not None:
        return backend
    with _REGISTRY_LOCK:
        lock = _LOAD_LOCKS.setdefault(key, threading.Lock())
    with lock:
        backend = _BACKENDS.get(key)
        if backend is None:
            backend = _construct_backend(model_name, device)
            _BACKENDS[key] = backend
    return backend


# ---- tokenizer and window: config files only, never the weights ------------

_TOKENIZERS: dict[str, object | None] = {}  # model -> tokenizers.Tokenizer, None = unavailable
_MAX_SEQ: dict[str, int] = {}
_TOK_LOCK = threading.Lock()

# Kana, CJK ideographs, Hangul: BERT-style tokenizers give each character its own token.
_CJK = "".join(
    f"{chr(lo)}-{chr(hi)}"
    for lo, hi in (
        (0x3040, 0x30FF),
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xAC00, 0xD7AF),
        (0xF900, 0xFAFF),
    )
)
# A BERT-style pre-tokenizer: CJK characters, word runs (no "_"), punctuation, "_".
_ESTIMATE_RE = re.compile(rf"[{_CJK}]|[^\W_{_CJK}]+|[^\w\s]|_")


def _estimate_tokens(text: str) -> int:
    """Conservative token count for when no tokenizer can be found.

    The chunker's :func:`sift.vault.chunk.estimate_tokens` is the project's one
    calibrated cost model (never under the bge-large count on ~10k windows, 1.2x
    margin), so the embedder defers to it: a passage sized by the chunker's estimate and
    counted here agree. :func:`_wordpiece_estimate` only covers that module failing to
    import.
    """
    try:
        from sift.vault.chunk import estimate_tokens
    except Exception:  # noqa: BLE001 - keep counting, whatever the chunker's state
        return _wordpiece_estimate(text)
    return int(estimate_tokens(text))


def _wordpiece_estimate(text: str) -> int:
    """Self-contained conservative estimate (fallback for :func:`_estimate_tokens`).

    Every punctuation mark and CJK character costs a token, a plain word one plus one
    per 4 characters, a long number one per 2 digits, and a digit/letter mix (hashes,
    base64, ids) or a very long run ~3 per 4 characters, because WordPiece shreds those.
    It errs high on purpose: over-counting only makes chunks smaller, under-counting
    silently truncates them (chars/4 counts a curl PoC at half its real size). Measured
    against the bge tokenizer: 1.0-1.2x on code, tables, hashes and CJK, ~1.7x on prose.
    """
    n = 0
    for m in _ESTIMATE_RE.finditer(text):
        word = m.group(0)
        if len(word) == 1:
            n += 1
        elif word.isdigit():
            n += 1 if len(word) <= 4 else math.ceil(len(word) / 2)
        elif len(word) > 24 or any(c.isdigit() for c in word):
            n += math.ceil(len(word) * 0.75)
        else:
            n += 1 + len(word) // 4
    return n


def _window_from_configs(configs: Iterable[dict | None]) -> int | None:
    """Smallest input window any of the config dicts declares, or None.

    Reads ``max_seq_length`` (sentence_bert_config.json: what sentence-transformers
    truncates at) and ``model_max_length`` / ``max_length`` (tokenizer_config.json:
    what fastembed truncates at). The smallest wins, so a passage fits either backend.
    """
    limits: list[int] = []
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        for key in ("max_seq_length", "model_max_length", "max_length"):
            value = cfg.get(key)
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            if 0 < value < _MAX_SANE_WINDOW:
                limits.append(int(value))
    return min(limits) if limits else None


def _local_model_dir(model_name: str) -> Path | None:
    """sentence-transformers also accepts a local directory as the model name."""
    try:
        p = Path(model_name).expanduser()
        return p if p.is_dir() else None
    except (OSError, ValueError):
        return None


def _hf_cached(repo_id: str, filename: str, cache_dir: str | None = None) -> Path | None:
    """A file from the local Hugging Face cache, without touching the network."""
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir)
    except Exception:  # noqa: BLE001 - not a hub id, no cache, ...
        return None
    return Path(hit) if isinstance(hit, str) and Path(hit).is_file() else None


def _fastembed_cached(model_name: str, filename: str) -> Path | None:
    """The same file in fastembed's cache: its ONNX export of the model, a different repo."""
    try:
        from fastembed import TextEmbedding
        from fastembed.common.utils import define_cache_dir

        desc = next(
            (
                m
                for m in TextEmbedding.list_supported_models()
                if str(m.get("model", "")).lower() == model_name.lower()
            ),
            None,
        )
        if desc is None:
            return None
        cache = str(define_cache_dir(None))
        hf_source = (desc.get("sources") or {}).get("hf")
        if hf_source and (hit := _hf_cached(hf_source, filename, cache_dir=cache)):
            return hit
        # Older exports arrive as a tarball unpacked to <cache>/fast-<name>/ or <cache>/<name>/.
        base = str(desc.get("model") or model_name).split("/")[-1]
        for d in (Path(cache) / f"fast-{base}", Path(cache) / base):
            if (d / filename).is_file():
                return d / filename
    except Exception:  # noqa: BLE001 - fastembed missing or its internals moved
        return None
    return None


def _hf_download(repo_id: str, filename: str) -> Path | None:
    """Fetch one small file from the Hub (honours HF_HUB_OFFLINE). None on any failure."""
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id, filename))
    except Exception as exc:  # noqa: BLE001
        log.debug("could not fetch %s for %s: %s", filename, repo_id, exc)
        return None


def _find_model_file(model_name: str, filename: str, *, download: bool = False) -> Path | None:
    """Locate one small file of ``model_name`` without loading the model. Never raises.

    Order: a local model directory; the Hugging Face cache (where sentence-transformers
    keeps the original repo); fastembed's cache. With ``download``, a miss is fetched
    from the original repo.
    """
    local = _local_model_dir(model_name)
    if local is not None:
        p = local / filename
        return p if p.is_file() else None
    hit = _hf_cached(model_name, filename) or _fastembed_cached(model_name, filename)
    if hit is None and download:
        hit = _hf_download(model_name, filename)
    return hit


def _read_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _load_tokenizer(model_name: str):
    path = _find_model_file(model_name, "tokenizer.json", download=True)
    if path is None:
        reason = "no tokenizer.json in any local cache and none could be downloaded"
    else:
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(path))
            # A tokenizer.json can carry truncation/padding (fastembed even switches
            # truncation on at the window); either would falsify every count.
            tok.no_truncation()
            tok.no_padding()
            return tok
        except Exception as exc:  # noqa: BLE001
            reason = f"{path} is unreadable ({type(exc).__name__}: {exc})"
    _warn_once(
        f"tokenizer:{model_name}",
        "token counts for %s are conservative estimates: %s",
        model_name,
        reason,
    )
    return None


def _tokenizer_for(model_name: str):
    if model_name in _TOKENIZERS:
        return _TOKENIZERS[model_name]
    with _TOK_LOCK:
        if model_name not in _TOKENIZERS:
            _TOKENIZERS[model_name] = _load_tokenizer(model_name)
        return _TOKENIZERS[model_name]


def _max_seq_for(model_name: str) -> int:
    window = _MAX_SEQ.get(model_name)
    if window is None:
        with _TOK_LOCK:
            window = _MAX_SEQ.get(model_name)
            if window is None:
                configs = [
                    _read_json(_find_model_file(model_name, name))
                    for name in ("sentence_bert_config.json", "tokenizer_config.json")
                ]
                window = _window_from_configs(configs) or _KNOWN_MAX_SEQ.get(
                    model_name.lower(), DEFAULT_MAX_SEQ_LENGTH
                )
                _MAX_SEQ[model_name] = window
    return window


# ---- the embedder ------------------------------------------------------------


class Embedder:
    def __init__(self, model_name: str, device: str = "auto", *, query_device: str | None = None):
        self.model_name = model_name
        self.query_prefix, self.passage_prefix = _prefixes_for(model_name)
        self._requested_device = device
        # None: queries share the passage backend (the historical behaviour).
        self._requested_query_device = (query_device or "").strip() or None
        self._device: str | None = None
        self._query_device: str | None = None
        self._lock = threading.Lock()  # lazy device resolution
        self._query_lock = threading.Lock()  # lazy query-backend selection
        self._backend: _Backend | None = None  # lazy
        self._query_backend: _Backend | None = None  # lazy

    # ---- devices ---------------------------------------------------------
    @property
    def device(self) -> str:
        """Resolved passage device, ``cuda`` or ``cpu``. Resolving ``auto`` imports torch."""
        if self._device is None:
            with self._lock:
                if self._device is None:
                    self._device = _resolve_device(self._requested_device)
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        with self._lock:
            self._device = value
            self._backend = None

    @property
    def query_device(self) -> str:
        """Device queries are embedded on: :attr:`device` unless a query device is set."""
        if self._requested_query_device is None:
            return self.device
        if self._query_device is None:
            with self._lock:
                if self._query_device is None:
                    self._query_device = _resolve_device(
                        self._requested_query_device, setting="SIFT_QUERY_DEVICE"
                    )
        return self._query_device

    # ---- backends --------------------------------------------------------
    def _load(self) -> _Backend:
        """The passage backend; it also serves queries unless a query device is set."""
        backend = self._backend
        if backend is None:
            backend = _load_backend(self.model_name, self.device)
            self._backend = backend
        return backend

    def _load_query(self) -> _Backend:
        if self._requested_query_device is None:
            return self._load()
        backend = self._query_backend
        if backend is not None:
            return backend
        with self._query_lock:
            if self._query_backend is None:
                device = self.query_device
                try:
                    self._query_backend = _load_backend(self.model_name, device)
                except Exception as exc:  # noqa: BLE001 - degrade, never fail a search
                    _warn_once(
                        f"query-backend:{self.model_name}:{device}",
                        "query backend for %s on %s failed to load (%s: %s); "
                        "embedding queries with the passage backend",
                        self.model_name,
                        device,
                        type(exc).__name__,
                        exc,
                    )
                    self._query_backend = self._load()
            return self._query_backend

    def _encode(
        self, texts: list[str], *, batch_size: int, kind: str = "passage"
    ) -> list[list[float]]:
        if not texts:
            return []
        backend = self._load_query() if kind == "query" else self._load()
        with backend.lock:
            if backend.kind == "st":
                arr = backend.model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=len(texts) > 512,
                )
                return [row.tolist() for row in arr]
            return [vec.tolist() for vec in backend.model.embed(texts, batch_size=batch_size)]

    # ---- public ----------------------------------------------------------
    def embed(
        self,
        texts: Sequence[str] | Iterable[str],
        *,
        kind: str = "passage",
        batch_size: int = 64,
    ) -> list[list[float]]:
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        return self._encode([prefix + t for t in texts], batch_size=batch_size, kind=kind)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], kind="query")[0]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text], kind="passage")[0]

    # ---- token budget ----------------------------------------------------
    def count_tokens(self, text: str) -> int:
        """Model tokens in ``text``: special tokens excluded, never truncated.

        Reads only the model's ``tokenizer.json`` (once per process), never the
        weights. Without one it returns a conservative estimate and warns once.
        """
        if not text:
            return 0
        tok = _tokenizer_for(self.model_name)
        if tok is None:
            return _estimate_tokens(text)
        return len(tok.encode(text, add_special_tokens=False).ids)

    @property
    def max_seq_length(self) -> int:
        """The model's hard input window in tokens, special tokens included.

        The smaller of sentence_bert_config.json's ``max_seq_length`` and
        tokenizer_config.json's ``model_max_length``; failing both, a table of known
        models, then 512. A ceiling, not a target chunk size.
        """
        return _max_seq_for(self.model_name)

    @property
    def num_special_tokens(self) -> int:
        """Special tokens added around one sequence ([CLS] ... [SEP] is 2)."""
        tok = _tokenizer_for(self.model_name)
        post = getattr(tok, "post_processor", None) if tok is not None else None
        if post is not None:
            try:
                return int(post.num_special_tokens_to_add(False))
            except Exception:  # noqa: BLE001
                pass
        return 2

    def passage_budget(self, header: str = "", *, margin: int = 2) -> int:
        """Tokens left for chunk text in one passage.

        ``header`` is what the caller puts before the chunk text (the pipeline's
        title, newline, heading, newline). Subtracts it, :attr:`passage_prefix`, the
        special tokens and ``margin`` tokens of slack for merges at the joins from
        :attr:`max_seq_length`. Can be <= 0 for a huge header; callers clamp.
        """
        used = self.count_tokens(self.passage_prefix + header)
        return self.max_seq_length - self.num_special_tokens - used - max(0, margin)


# ---- process-wide singleton ---------------------------------------------------

_BUILD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _build_embedder(model_name: str, device: str, query_device: str | None) -> Embedder:
    return Embedder(model_name, device, query_device=query_device)


def get_embedder() -> Embedder:
    """The process-wide :class:`Embedder` for the current settings (thread-safe)."""
    s = get_settings()
    query_device = str(getattr(s, "query_device", "") or "").strip() or None
    with _BUILD_LOCK:  # lru_cache alone may run the builder once per racing thread
        return _build_embedder(s.embed_model, s.embed_device, query_device)


def _reset() -> None:
    """Forget the singleton, loaded models, tokenizers and one-time warnings."""
    with _BUILD_LOCK:
        _build_embedder.cache_clear()
    with _REGISTRY_LOCK:
        _BACKENDS.clear()
        _LOAD_LOCKS.clear()
    with _TOK_LOCK:
        _TOKENIZERS.clear()
        _MAX_SEQ.clear()
    with _WARNED_LOCK:
        _WARNED.clear()


# tests/conftest.py resets the singleton between tests through this attribute.
get_embedder.cache_clear = _reset  # type: ignore[attr-defined]


def warmup(*, query: bool = True, passage: bool = False) -> bool:
    """Load the model(s) now so the first real call skips the cold start.

    Meant for a daemon thread started with the MCP server: loads are single-flight,
    so a tool call that arrives mid-warm-up waits for this load rather than starting
    a second copy. ``query`` loads whatever serves :meth:`Embedder.embed_query` (the
    query backend, if ``SIFT_QUERY_DEVICE`` is set). ``passage`` also loads the
    passage backend and tokenizer; on cuda that holds ~1.3 GB of VRAM for the life
    of the process, even if the session never writes a note.

    Never raises: a failure is logged and the lazy path stays in place. Returns
    True when everything requested loaded.
    """
    try:
        emb = get_embedder()
        if query:
            emb.embed_query("warm-up")
        if passage:
            emb.embed_one("warm-up")
            count = getattr(emb, "count_tokens", None)
            if count is not None:
                count("warm-up")
        return True
    except Exception as exc:  # noqa: BLE001 - a warm-up must never take the server down
        log.warning(
            "embedder warm-up failed (%s: %s); the model will load on first use",
            type(exc).__name__,
            exc,
        )
        return False
