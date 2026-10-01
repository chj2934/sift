"""LanceDB-backed hybrid search over note chunks.

We run a vector query and a full-text (BM25) query separately and fuse them with
Reciprocal Rank Fusion. Doing the fusion ourselves (rather than leaning on
LanceDB's built-in ``query_type="hybrid"``) keeps behaviour stable across
LanceDB versions and makes it unit-testable without a GPU or network.

Write model. Every LanceDB write is a commit: a new table version plus a manifest
that lists every fragment. One commit per note grew the real index to 12,920
fragments, 53k versions and 29 GB for ~1 GB of data, and every query paid for the
fragment count. So:

* batch writes are one commit (`delete_notes`, `replace_notes`);
* the FTS index is created once (`ensure_fts`) and kept current by `optimize`,
  never rebuilt per write - FTS queries already scan rows added since the build;
* `optimize` compacts and prunes old versions, but never versions younger than a
  safety window, because a running MCP server may still be reading them.

This module never prints: it is reachable from the MCP server, whose stdout is the
JSON-RPC wire. Diagnostics go to the ``sift.index.store`` logger.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sift.config import get_settings
from sift.index import embed as _embed

if TYPE_CHECKING:
    import pyarrow as pa

log = logging.getLogger(__name__)

TABLE = "notes"
RRF_K = 60
_YEAR_SECS = 31_557_600.0

# Literals per `IN (...)` list. Lists are OR-ed into one predicate, so a batch is
# still a single commit; this only bounds the size of each list.
_IN_CHUNK = 500
# Ids per commit. A 21k-literal predicate parsed and ran in 0.13 s on lancedb 0.38,
# so any realistic batch (a flush, a prune plan) is one commit.
_MAX_PER_COMMIT = 10_000
# Ceiling for the one-off pool widening in `search`.
_MAX_POOL = 400
# A note's score is its best chunk's RRF score plus this share of each further matched
# chunk's - for its strongest few extra chunks only. Uncapped, the bonus grew with the
# number of pieces a note was cut into, so chunks sized to the 512-token embedder
# window (more, smaller chunks than the old ~800-token ones) would have lifted long
# notes on length alone, and a long note's chunks already crowd the candidate pool.
_EXTRA_CHUNK_CREDIT = 0.25
_MAX_EXTRA_CHUNKS = 3

# 'CWE-79', 'cwe-79', '79', 'CWE 79', 'CWE_79', 'CWE79' all mean CWE-79.
_CWE_NUM_RE = re.compile(r"(?:CWE)?[-_ ]?(\d+)")
# Any other filter token must be dash-joined [A-Z0-9] words (e.g. NVD-CWE-OTHER): no
# quotes, none of LIKE's wildcards (% and _), and no dangling "CWE-".
_CWE_WORD_RE = re.compile(r"[A-Z0-9]+(?:-[A-Z0-9]+)*")
_LIST_SPLIT_RE = re.compile(r"[,;\s]+")


class IndexDimMismatch(RuntimeError):
    """The table's vectors have a different width than the configured embedder."""


def norm_path(path: str | os.PathLike[str] | None) -> str:
    """Canonical spelling of a path for comparisons: absolute, and case/separator
    normalised on Windows. Stored `path` values are kept as written, so compare with
    this, but pass the stored spelling back to `delete_paths`."""
    if not path:
        return ""
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _sql_str(value: object) -> str:
    """Body of a SQL string literal: single quotes doubled, never stripped, so a
    value like O'Brien still matches itself."""
    return str(value).replace("'", "''")


def _in_predicate(column: str, values: list[str]) -> str:
    parts = []
    for i in range(0, len(values), _IN_CHUNK):
        literals = ",".join("'" + _sql_str(v) + "'" for v in values[i : i + _IN_CHUNK])
        parts.append(f"{column} IN ({literals})")
    return parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"


def _batches(values: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _distinct(values: Iterable[str] | None) -> list[str]:
    """Order-preserving de-duplication that drops empty values."""
    if values is None:
        return []
    return list(dict.fromkeys(str(v) for v in values if v))


def _cwe_tokens(items: Iterable[str]) -> list[str]:
    """Stored form of a CWE list: one upper-case token per CWE.

    `_normalize_cwe` only strips and upper-cases each item, so a hand-written
    ``cwe: CWE-79, CWE-89`` arrives as one item. Splitting here keeps the
    token-exact filter in `_cwe_clause` matching it."""
    out: list[str] = []
    for item in items:
        for tok in _LIST_SPLIT_RE.split(str(item).strip().upper()):
            if not tok:
                continue
            out.append(f"CWE-{tok}" if tok.isdigit() else tok)
    return out


def _cwe_clause(value: object) -> str:
    """Exact-token CWE match. ``LIKE '%CWE-20%'`` also matched CWE-200 and CWE-209,
    and CWE-78 matched CWE-787; padding both sides with spaces makes it a token."""
    raw = str(value).strip().upper()
    m = _CWE_NUM_RE.fullmatch(raw)
    if m:
        token = f"CWE-{m.group(1)}"
    elif _CWE_WORD_RE.fullmatch(raw):
        token = raw
    else:
        raise ValueError(f"cwe filter must look like 'CWE-79' or '79', got {value!r}")
    return f"(' ' || cwe_str || ' ') LIKE '% {token} %'"


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


@lru_cache(maxsize=8)
def _connect(path: str):
    """One connection per database directory, shared by every Store in the process.

    ``read_consistency_interval=0`` makes every read check for a newer table version,
    so a cached table handle still sees writes made through other handles and other
    processes (a CLI ingest while the MCP server runs, remember-then-search) and
    follows a drop-and-recreate. Measured on local disk: no cost. A non-zero interval
    served stale rows to the next search, and a handle with the default interval kept
    returning a dropped table's rows.
    """
    import lancedb  # deferred: keeps `import sift.mcp_server` cheap

    return lancedb.connect(path, read_consistency_interval=timedelta(0))


def _quality_mult(quality: int) -> float:
    """Map a 0-100 quality score to a ranking multiplier (~0.6 .. 1.4 at weight 1.0)."""
    w = get_settings().quality_weight
    if w <= 0:
        return 1.0
    return 1.0 + w * (0.8 * (quality / 100.0) - 0.4)


def _recency_mult(created_ts: float) -> float:
    """Mild boost for recent notes, mild decay for old ones (~0.75 .. 1.15 at weight 1.0).

    Unknown dates (created_ts == 0) are neutral.
    """
    w = get_settings().recency_weight
    if w <= 0 or not created_ts:
        return 1.0
    age_years = max(0.0, (time.time() - created_ts) / _YEAR_SECS)
    delta = max(-0.25, min(0.15, 0.15 - 0.06 * age_years))
    return 1.0 + w * delta


def _schema(dim: int) -> pa.Schema:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("id", pa.string()),  # chunk id, see ChunkRow.chunk_id
            pa.field("note_id", pa.string()),
            pa.field("slug", pa.string()),
            pa.field("type", pa.string()),
            pa.field("title", pa.string()),
            pa.field("heading", pa.string()),
            pa.field("text", pa.string()),  # excerpt shown to the user
            pa.field("search_text", pa.string()),  # title + heading + text, for BM25
            pa.field("chunk_index", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("source", pa.string()),
            pa.field("url", pa.string()),
            pa.field("cwe_str", pa.string()),  # space-joined tokens, for the CWE filter
            pa.field("tags_str", pa.string()),
            pa.field("severity", pa.string()),
            pa.field("program", pa.string()),
            pa.field("path", pa.string()),
            pa.field("mtime", pa.float64()),
            pa.field("quality", pa.int32()),  # 0-100 heuristic retrieval-worth
            # The note's disclosed/created date, epoch seconds (0 = unknown).
            pa.field("created_ts", pa.float64()),
        ]
    )


@dataclass
class ChunkRow:
    note_id: str
    slug: str
    type: str
    title: str
    heading: str
    text: str
    chunk_index: int
    vector: list[float]
    source: str = ""
    url: str = ""
    cwe: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    severity: str = ""
    program: str = ""
    path: str = ""
    mtime: float = 0.0
    quality: int = 50
    created_ts: float = 0.0

    @property
    def chunk_id(self) -> str:
        """Unique per FILE, not just per note id.

        Two vault files can share a frontmatter id (KEV and NVD records of one CVE,
        two long titles truncated to the same id). With ``{note_id}::{index}`` their
        chunks collided: RRF credited one chunk id twice, and a merge_insert batch
        holding both was rejected as ambiguous. Rows written before this format keep
        their old ids until rewritten; nothing parses the id.
        """
        if self.path:
            tag = hashlib.blake2s(norm_path(self.path).encode("utf-8"), digest_size=4).hexdigest()
            return f"{self.note_id}::{tag}::{self.chunk_index}"
        return f"{self.note_id}::{self.chunk_index}"

    def to_record(self) -> dict:
        parts = [self.title, self.heading, self.text, " ".join(self.tags), " ".join(self.cwe)]
        return {
            "id": self.chunk_id,
            "note_id": self.note_id,
            "slug": self.slug,
            "type": self.type,
            "title": self.title,
            "heading": self.heading,
            "text": self.text,
            "search_text": "\n".join(p for p in parts if p),
            "chunk_index": self.chunk_index,
            "vector": self.vector,
            "source": self.source,
            "url": self.url,
            "cwe_str": " ".join(_cwe_tokens(self.cwe)),
            "tags_str": " ".join(self.tags),
            "severity": self.severity or "",
            "program": self.program or "",
            "path": self.path,
            "mtime": self.mtime,
            "quality": int(self.quality),
            "created_ts": float(self.created_ts),
        }


def _last_copy_per_file(rows: list[ChunkRow]) -> list[ChunkRow]:
    """Drop superseded copies when one batch holds the same file twice.

    A source can yield the same note twice in one flush. Its rows then repeat chunk
    ids, which merge_insert rejects ("ambiguous merge insert") when they already
    exist and silently inserts twice when they do not. The later copy is what is on
    disk, so it wins.

    A file is keyed the way its chunk ids are (note id + canonical path), so two
    spellings of one Windows path are one file. A new copy of a file starts when one
    of its chunk indexes repeats; rows of different files may interleave freely.
    """
    current: dict[tuple[str, str], tuple[set[int], list[ChunkRow]]] = {}
    for r in rows:
        key = (r.note_id, norm_path(r.path))
        entry = current.get(key)
        if entry is None or r.chunk_index in entry[0]:
            entry = current[key] = (set(), [])  # first or fresh copy; an earlier one is superseded
        entry[0].add(r.chunk_index)
        entry[1].append(r)
    keep = {id(r) for _idx, copy in current.values() for r in copy}
    if len(keep) == len(rows):
        return rows
    return [r for r in rows if id(r) in keep]


@dataclass
class Hit:
    note_id: str
    slug: str
    type: str
    title: str
    url: str
    source: str
    severity: str
    program: str
    path: str
    score: float
    excerpt: str
    heading: str
    matched_chunks: int
    quality: int = 0
    created_ts: float = 0.0
    #: Indices (into the searched phrasings) of the queries that found this note.
    matched_queries: list[int] = field(default_factory=list)


#: Upper bound on phrasings per multi-query search: each one costs a vector and a
#: keyword search, and past a handful they stop adding new notes.
MAX_QUERIES = 6


def normalize_queries(query: str | None, queries: Iterable[str] | None = None) -> list[str]:
    """`query` followed by `queries`, stripped, blanks dropped, and repeats removed
    case-insensitively (the first spelling wins). Order is preserved: index 0 is the
    primary phrasing, which the reranker scores against."""
    out: list[str] = []
    seen: set[str] = set()
    for q in [query, *(queries or [])]:
        q = (q or "").strip()
        if q and q.casefold() not in seen:
            seen.add(q.casefold())
            out.append(q)
    return out


class Store:
    #: Routine `optimize()` compacts only past this many fragments. A table under 1M
    #: rows is all "small fragments" to Lance, so compaction rewrites the whole table
    #: once earlier fragments are indexed (measured: a full 85 MB copy per call on a
    #: 20k x 1024-dim table, kept on disk for the retention window). Fragments cost
    #: ~0.07 ms each per query, so 64 is noise; 12,920 was ~1 s.
    COMPACT_MIN_FRAGMENTS = 64
    #: ... or once this many rows are missing from the FTS index (each keyword query
    #: flat-scans them).
    COMPACT_MIN_UNINDEXED = 2_000
    #: Above this a routine call skips and asks for `sift compact`: the first cleanup
    #: of a long-neglected table takes minutes, which belongs in an explicit user step.
    COMPACT_MAX_ROUTINE_FRAGMENTS = 2_000
    #: Never prune versions younger than this. A reader in another process (the MCP
    #: server) may still be reading them; a retention of 0 was measured to make such a
    #: reader fail with "Not found".
    MIN_RETAIN = timedelta(minutes=10)
    DEFAULT_RETAIN = timedelta(hours=1)

    def __init__(self, db_path: Path | None = None, dim: int | None = None):
        s = get_settings()
        self.db_path = Path(db_path) if db_path else s.resolved_db()
        self.dim = dim or s.effective_embed_dim()
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._tbl = None
        # Why the last search() degraded (e.g. keyword search unavailable). Empty when
        # it ran clean. Callers surface these instead of passing off a partial result
        # as the whole answer.
        self.warnings: list[str] = []

    # ---- lifecycle ----------------------------------------------------
    @property
    def _db(self):
        if self._conn is None:
            self._conn = _connect(str(self.db_path.resolve()))
        return self._conn

    def _table_names(self) -> list[str]:
        try:
            return list(self._db.list_tables().tables)  # lancedb >= 0.30
        except AttributeError:
            return list(self._db.table_names())

    def _open(self, *, create: bool):
        if TABLE in self._table_names():
            tbl = self._db.open_table(TABLE)
        elif create:
            tbl = self._db.create_table(TABLE, schema=_schema(self.dim), exist_ok=True)
        else:
            return None
        self._check_dim(tbl)
        return tbl

    def table(self):
        """The notes table, created if missing. Cached on this Store; the strongly
        consistent connection keeps the cached handle current."""
        if self._tbl is None:
            self._tbl = self._open(create=True)
        return self._tbl

    def _existing_table(self):
        """The notes table, or None if there is none. Reads never create the table."""
        if self._tbl is None:
            self._tbl = self._open(create=False)
        return self._tbl

    def _check_dim(self, tbl) -> None:
        try:
            existing = tbl.schema.field("vector").type.list_size
        except Exception:  # noqa: BLE001
            return
        if existing and existing != self.dim:
            raise IndexDimMismatch(
                f"index was built with {existing}-dim vectors but the current model "
                f"produces {self.dim}-dim. Run `sift reindex --force` after changing SIFT_EMBED_MODEL."
            )

    def drop(self) -> None:
        self._tbl = None  # a handle to the dropped table would write into the void
        if TABLE in self._table_names():
            self._db.drop_table(TABLE)

    # ---- full-text index and maintenance -------------------------------
    @staticmethod
    def _fts_index(tbl):
        """The FTS index on search_text, or None. AttributeError on a lancedb
        without list_indices()."""
        for ix in tbl.list_indices():
            kind = str(getattr(ix, "index_type", "")).upper()
            if kind in {"FTS", "INVERTED"} and list(getattr(ix, "columns", None) or []) == [
                "search_text"
            ]:
                return ix
        return None

    def has_fts(self) -> bool:
        try:
            tbl = self._existing_table()
            return tbl is not None and self._fts_index(tbl) is not None
        except Exception:  # noqa: BLE001
            return False

    def ensure_fts(self, *, force: bool = False) -> bool:
        """Create the BM25 index on search_text if it is missing; ``force`` rebuilds it.

        Ordinary writes do not need this: FTS queries also scan rows added since the
        build and skip deleted ones, and `optimize` folds new rows in. A rebuild
        re-tokenises every row and leaves another index dir behind, which is why this
        no longer replaces by default. Creates the table if needed (after `drop`, this
        is what restores keyword search). Returns whether the index exists afterwards;
        never raises - a failed build degrades keyword search, it loses no data.
        """
        tbl = None
        try:
            tbl = self.table()
            exists: bool | None = None
            if not force:
                try:
                    exists = self._fts_index(tbl) is not None
                except AttributeError:
                    exists = None  # older lancedb: cannot tell, so build as it always did
                if exists:
                    return True
            replace = force or exists is None
            try:
                from lancedb.index import FTS

                tbl.create_index("search_text", config=FTS(), replace=replace)
            except (TypeError, ImportError):
                tbl.create_fts_index("search_text", replace=replace)  # older lancedb
            return True
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                if tbl is not None and self._fts_index(tbl) is not None:
                    return True  # a concurrent builder won the race
            log.warning("FTS index build failed (%s); keyword search degraded", exc)
            return False

    def disk_bytes(self) -> int:
        """Bytes on disk for the table, every retained version included. Walks the
        table directory - fine for a CLI report, not for a hot path."""
        total = 0
        for root, _dirs, files in os.walk(self.db_path / f"{TABLE}.lance"):
            for name in files:
                with contextlib.suppress(OSError):
                    total += os.path.getsize(os.path.join(root, name))
        return total

    def _info(self, tbl) -> dict:
        st = tbl.stats()
        frag = _get(st, "fragment_stats") or {}
        info: dict[str, Any] = {
            "rows": _get(st, "num_rows"),
            "fragments": _get(frag, "num_fragments"),
            "data_bytes": _get(st, "total_bytes"),
            "version": tbl.version,
            "fts": False,
            "fts_unindexed_rows": None,
        }
        with contextlib.suppress(Exception):
            ix = self._fts_index(tbl)
            if ix is not None:
                info["fts"] = True
                ixs = tbl.index_stats(ix.name)
                info["fts_unindexed_rows"] = _get(ixs, "num_unindexed_rows")
        return info

    def index_info(self, *, disk: bool = False) -> dict:
        """Row/fragment/version counts and FTS coverage, for `sift status`/`compact`.
        ``disk=True`` adds `disk_bytes` (walks the table dir)."""
        tbl = self._existing_table()
        info = self._info(tbl) if tbl is not None else {"rows": 0, "fragments": 0, "version": 0}
        info["exists"] = tbl is not None
        if disk:
            info["disk_bytes"] = self.disk_bytes()
        return info

    def optimize(
        self,
        retain: timedelta | None = None,
        *,
        force: bool = False,
        rebuild_fts: bool = False,
        measure_disk: bool = False,
        allow_unsafe_retain: bool = False,
    ) -> dict:
        """Compact fragments, fold new rows into the FTS index, and prune versions
        older than ``retain`` (default 1 h). Then make sure the FTS index exists - it
        is missing after `drop` - or rebuild it once when ``rebuild_fts``.

        Call it once at the end of a CLI ingest or reindex, never per note and never
        from an MCP tool. A routine call compacts only when the fragment backlog or the
        unindexed FTS rows pass a threshold (see the class constants), because each
        compaction rewrites the whole table. ``force=True`` (``sift compact``) always
        compacts. Versions younger than ``retain`` survive, so a concurrent reader
        keeps working; a retention under 10 minutes needs ``allow_unsafe_retain`` and
        every other sift process, MCP servers included, stopped.

        Never raises for a maintenance failure (the data is already committed): the
        error is logged and returned in the report. Raises ValueError for an unsafe
        ``retain``.
        """
        retain = self.DEFAULT_RETAIN if retain is None else retain
        if retain < self.MIN_RETAIN and not allow_unsafe_retain:
            raise ValueError(
                f"retain={retain} is under {self.MIN_RETAIN}: a running MCP server may still be "
                "reading those versions. Pass allow_unsafe_retain=True only with every sift "
                "process stopped."
            )
        report: dict[str, Any] = {
            "ran": False,
            "reason": "",
            "error": None,
            "seconds": 0.0,
            "before": {},
            "after": {},
        }
        try:
            tbl = self._existing_table()
        except Exception as exc:  # noqa: BLE001 - e.g. a dimension mismatch
            log.warning("index maintenance skipped: %s", exc)
            report["error"] = str(exc)
            return report
        if tbl is None:
            report["reason"] = "no index yet"
            return report

        t0 = time.monotonic()
        try:
            before = self._info(tbl)
            if measure_disk:
                before["disk_bytes"] = self.disk_bytes()
            report["before"] = before
            fragments = int(before.get("fragments") or 0)
            unindexed = int(before.get("fts_unindexed_rows") or 0)
            run = force
            if force:
                report["reason"] = "forced"
            elif fragments > self.COMPACT_MAX_ROUTINE_FRAGMENTS:
                report["reason"] = f"{fragments} fragments: run `sift compact` once"
                log.warning(
                    "index has %d fragments; skipping routine compaction. Run `sift compact` "
                    "once (it takes minutes; a running MCP server is unaffected).",
                    fragments,
                )
            elif fragments >= self.COMPACT_MIN_FRAGMENTS:
                run, report["reason"] = True, f"{fragments} fragments"
            elif unindexed >= self.COMPACT_MIN_UNINDEXED:
                run, report["reason"] = True, f"{unindexed} rows outside the FTS index"
            else:
                report["reason"] = "below thresholds"
            if run:
                log.info(
                    "optimizing index (%d fragments, version %s)...",
                    fragments,
                    before.get("version"),
                )
                tbl.optimize(cleanup_older_than=retain)
                report["ran"] = True
        except Exception as exc:  # noqa: BLE001
            log.warning("index maintenance failed (%s); the data is committed and searchable", exc)
            report["error"] = str(exc)

        if not self.ensure_fts(force=rebuild_fts) and report["error"] is None:
            report["error"] = "FTS index build failed"
        with contextlib.suppress(Exception):
            after = self._info(tbl)
            if measure_disk:
                after["disk_bytes"] = self.disk_bytes()
            report["after"] = after
        report["seconds"] = round(time.monotonic() - t0, 3)
        if report["ran"]:
            log.info(
                "index optimized: %s -> %s fragments, version %s -> %s (%.1fs)",
                report["before"].get("fragments"),
                report["after"].get("fragments"),
                report["before"].get("version"),
                report["after"].get("version"),
                report["seconds"],
            )
        return report

    def count(self) -> int:
        try:
            tbl = self._existing_table()
            return tbl.count_rows() if tbl is not None else 0
        except Exception:  # noqa: BLE001
            return 0

    # ---- what is indexed ----------------------------------------------
    def _scan(self, columns: list[str]):
        """Selected columns of every row, as Arrow; None when there is no table."""
        tbl = self._existing_table()
        if tbl is None:
            return None
        return tbl.search().select(columns).limit(0).to_arrow()  # limit(0) = every row

    def indexed_files(self) -> list[tuple[str, str, float]]:
        """Distinct ``(note_id, path, mtime)`` per indexed file, from one scan of three
        columns. Unlike `note_index`, two files sharing an id both appear. ``path`` is
        the stored spelling ('' when unknown); compare it through `norm_path`."""
        try:
            at = self._scan(["note_id", "path", "mtime"])
        except Exception as exc:  # noqa: BLE001 - e.g. an old table without the columns
            log.warning("could not read the indexed notes (%s); treating the index as empty", exc)
            return []
        if at is None:
            return []
        out: dict[tuple[str, str], float] = {}
        for nid, path, mtime in zip(
            at.column("note_id").to_pylist(),
            at.column("path").to_pylist(),
            at.column("mtime").to_pylist(),
            strict=True,
        ):
            if nid is not None:
                # Chunks of one file share its mtime; last row wins.
                out[(nid, path or "")] = float(mtime or 0.0)
        return [(nid, path, mtime) for (nid, path), mtime in out.items()]

    def note_index(self) -> dict[str, tuple[float, str]]:
        """note_id -> (indexed mtime, stored path). When files share an id the last row
        wins; use `indexed_files` to see every file."""
        return {nid: (mtime, path) for nid, path, mtime in self.indexed_files()}

    def note_mtimes(self) -> dict[str, float]:
        """note_id -> indexed mtime, for skipping unchanged notes on reindex."""
        return {nid: mtime for nid, (mtime, _path) in self.note_index().items()}

    def path_state(self) -> dict[str, tuple[str, float]]:
        """Stored path -> (note_id, indexed mtime), for path-keyed syncing. Rows with
        no path are left out."""
        return {path: (nid, mtime) for nid, path, mtime in self.indexed_files() if path}

    def indexed_ids(self) -> set[str]:
        return {nid for nid, _path, _mtime in self.indexed_files()}

    def stored_paths(self, note_ids: Iterable[str]) -> list[str]:
        """Distinct stored paths (their stored spelling, '' for rows without one) of
        every row carrying one of these ids, in row order. A targeted read: only the
        matching rows are scanned. Never creates the table; raises on a read failure."""
        ids = _distinct(note_ids)
        tbl = self._existing_table()
        if not ids or tbl is None:
            return []
        out: dict[str, None] = {}
        for group in _batches(ids, _MAX_PER_COMMIT):
            where = _in_predicate("note_id", group)
            n = tbl.count_rows(where)
            if not n:
                continue
            at = tbl.search().where(where).select(["path"]).limit(n).to_arrow()
            for p in at.column("path").to_pylist():
                out.setdefault(p or "", None)
        return list(out)

    # ---- writes -----------------------------------------------------------
    def _delete_matching(self, predicate: str) -> int:
        tbl = self._existing_table()
        if tbl is None:
            return 0
        # A delete that matches nothing still commits a version, and most ids in an
        # ingest flush are new. Counting first costs a scan but no commit.
        n = tbl.count_rows(predicate)
        if n:
            tbl.delete(predicate)
        return n

    def delete_notes(self, note_ids: Iterable[str]) -> int:
        """Delete every chunk of these notes in ONE commit (per 10,000 ids); ids with no
        rows cost no commit. Returns the number of rows deleted.

        Quotes are doubled, never stripped. Nothing is suppressed: a failed delete must
        stop the caller before it adds replacement rows, or the index ends up holding
        both copies.
        """
        ids = _distinct(note_ids)
        return sum(
            self._delete_matching(_in_predicate("note_id", g))
            for g in _batches(ids, _MAX_PER_COMMIT)
        )

    def delete_paths(self, paths: Iterable[str]) -> int:
        """`delete_notes`, keyed on the stored `path` column (pass stored spellings,
        e.g. from `path_state`). One commit per 10,000 paths."""
        ps = _distinct(paths)
        return sum(
            self._delete_matching(_in_predicate("path", g)) for g in _batches(ps, _MAX_PER_COMMIT)
        )

    def delete_note(self, note_id: str) -> None:
        """Delete one note's chunks. Errors propagate (they used to be swallowed, which
        let the caller add a second copy)."""
        self.delete_notes([note_id])

    def add_chunks(self, rows: list[ChunkRow]) -> None:
        """Plain append, one commit. Use `replace_notes` when the notes may already be
        indexed."""
        if not rows:
            return
        self.table().add([r.to_record() for r in rows])

    def replace_notes(
        self,
        rows: list[ChunkRow],
        note_ids: Iterable[str] | None = None,
        *,
        paths: Iterable[str] | None = None,
    ) -> int:
        """Make ``rows`` the only indexed chunks of ``note_ids`` and of ``paths``, in ONE
        atomic commit (a merge_insert), so a reader never sees the notes missing.

        ``note_ids`` defaults to the ids in ``rows``; list an id with no rows to clear a
        note whose body no longer yields chunks. ``paths`` defaults to the rows' paths,
        which also clears a file whose frontmatter id was edited. Pass ``note_ids=[]``
        to scope by path alone: a path-keyed reindex then re-indexes one of two files
        sharing an id (KEV and NVD twins of a CVE) without clearing the other's rows.
        Paths match the stored spelling exactly. If one batch holds the same file twice,
        the later copy wins. Returns the number of rows written. Raises on failure, and
        then nothing was applied.
        """
        rows = _last_copy_per_file(list(rows))
        ids = _distinct(note_ids if note_ids is not None else (r.note_id for r in rows))
        ps = _distinct(paths if paths is not None else (r.path for r in rows))
        clauses = [_in_predicate("note_id", ids)] if ids else []
        if ps:
            clauses.append(_in_predicate("path", ps))
        scope = " OR ".join(clauses)
        if not rows:
            if scope:
                self._delete_matching(scope)
            return 0
        records = [r.to_record() for r in rows]
        tbl = self.table()
        if not scope:
            tbl.add(records)
        else:
            (
                tbl.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(scope)
                .execute(records)
            )
        return len(records)

    def upsert_note(self, rows: list[ChunkRow], *, note_id: str | None = None) -> None:
        """Replace one note's chunks in a single commit. With no rows, pass ``note_id``
        to clear a note whose body no longer yields chunks."""
        nid = note_id or (rows[0].note_id if rows else None)
        if not nid:
            return
        self.replace_notes(rows, [nid])

    def rescore(self, scores: Mapping[str, int]) -> int:
        """Set the stored `quality` of already-indexed notes, without re-embedding.

        Only notes whose stored score differs are touched, one commit per distinct new
        score (lance's update has no CASE). Scores are clamped to 0-100. CLI-only; run
        `optimize` afterwards. Returns the number of notes changed.
        """
        tbl = self._existing_table()
        if tbl is None or not scores:
            return 0
        at = tbl.search().select(["note_id", "quality"]).limit(0).to_arrow()
        stored: dict[str, set[int]] = {}
        for nid, q in zip(
            at.column("note_id").to_pylist(), at.column("quality").to_pylist(), strict=True
        ):
            if nid is not None:
                stored.setdefault(nid, set()).add(int(q or 0))
        groups: dict[int, list[str]] = {}
        for nid, q in scores.items():
            want = max(0, min(100, int(q)))
            have = stored.get(nid)
            if have is not None and have != {want}:
                groups.setdefault(want, []).append(nid)
        changed = 0
        for want, ids in sorted(groups.items()):
            for group in _batches(sorted(ids), _MAX_PER_COMMIT):
                tbl.update(where=_in_predicate("note_id", group), values={"quality": want})
                changed += len(group)
        return changed

    # ---- search ---------------------------------------------------------
    @staticmethod
    def _where(filters: dict | None) -> str | None:
        """SQL prefilter for search. Raises ValueError on a malformed CWE filter, so a
        typo is reported instead of answering "nothing found"."""
        if not filters:
            return None
        clauses: list[str] = []
        if t := filters.get("type"):
            clauses.append(f"type = '{_sql_str(str(t).strip().lower())}'")
        if p := filters.get("program"):
            # Case-insensitive, like list_notes: "google" must find notes stored as "Google".
            clauses.append(f"lower(program) = '{_sql_str(str(p).strip().lower())}'")
        if c := filters.get("cwe"):
            clauses.append(_cwe_clause(c))
        if sev := filters.get("severity"):
            clauses.append(f"severity = '{_sql_str(str(sev).strip().lower())}'")
        return " AND ".join(clauses) if clauses else None

    def _run(
        self,
        make_query: Callable[[], Any],
        k: int,
        where: str | None,
        *,
        label: str,
        degrade: bool = False,
    ) -> list[dict]:
        """Run one retriever. The where clause is never dropped: a filtered query that
        fails must not come back unfiltered. Each mode is tried twice, to ride out a
        transient error such as a Windows sharing violation while another process
        commits. Then it raises, or with ``degrade`` logs, records a warning and
        returns []."""
        modes = ("prefilter", "postfilter") if where else ("plain",)
        last: Exception | None = None
        for attempt in range(2):
            if attempt:
                time.sleep(0.05)
            for mode in modes:
                try:
                    q = make_query().limit(k)
                    if mode == "prefilter":
                        q = q.where(where, prefilter=True)
                    elif mode == "postfilter":
                        q = q.where(where)
                    return q.to_list()
                except Exception as exc:  # noqa: BLE001
                    last = exc
        if degrade:
            log.warning("%s search failed: %s", label, last)
            self.warnings.append(f"{label} search unavailable: {last}")
            return []
        raise RuntimeError(f"{label} search failed: {last}") from last

    def _retrieve(self, tbl, vec: list[float], query: str, pool: int, where: str | None):
        vhits = self._run(lambda: tbl.search(vec), pool, where, label="vector")
        # Keyword search degrades instead of failing: a missing or broken FTS index
        # must not take a working vector search down with it.
        fhits = self._run(
            lambda: tbl.search(query, query_type="fts", fts_columns="search_text"),
            pool,
            where,
            label="fts",
            degrade=True,
        )
        return vhits, fhits

    @staticmethod
    def _fuse(vhits: list[dict], fhits: list[dict], min_quality: int) -> list[Hit]:
        """Fuse one phrasing's vector and keyword lists."""
        return Store._fuse_many([(0, vhits), (0, fhits)], min_quality)

    @staticmethod
    def _fuse_many(lists: list[tuple[int, list[dict]]], min_quality: int) -> list[Hit]:
        """Reciprocal Rank Fusion over chunk ids across every ranked list.

        Each phrasing contributes two lists (vector, keyword) tagged with its index, so a
        chunk found by several phrasings collects several RRF terms and rises - the point
        of searching more than one phrasing.
        """
        scores: dict[str, float] = {}
        chunk_by_id: dict[str, dict] = {}
        for _qi, hits in lists:
            seen: set[str] = set()
            for rank, row in enumerate(hits):
                cid = row["id"]
                if cid in seen:
                    # Rows written before chunk ids were per-file can repeat an id within
                    # one list; crediting it twice doubled that note's score.
                    continue
                seen.add(cid)
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
                chunk_by_id.setdefault(cid, row)

        # Collapse chunks -> notes.
        notes: dict[str, Hit] = {}
        for cid, sc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            row = chunk_by_id[cid]
            nid = row["note_id"]
            if nid in notes:
                hit = notes[nid]
                hit.matched_chunks += 1
                # Strongest first (sorted by score), so the credited extras are the best.
                if hit.matched_chunks <= 1 + _MAX_EXTRA_CHUNKS:
                    hit.score += sc * _EXTRA_CHUNK_CREDIT
                continue
            notes[nid] = Hit(
                note_id=nid,
                slug=row.get("slug", ""),
                type=row.get("type", ""),
                title=row.get("title", ""),
                url=row.get("url", "") or "",
                source=row.get("source", "") or "",
                severity=row.get("severity", "") or "",
                program=row.get("program", "") or "",
                path=row.get("path", "") or "",
                score=sc,
                excerpt=(row.get("text", "") or "")[:600],
                heading=row.get("heading", "") or "",
                matched_chunks=1,
                quality=int(row.get("quality") or 0),
                created_ts=float(row.get("created_ts") or 0.0),
            )

        results = list(notes.values())
        if min_quality > 0:
            # The threshold is also in the where clause; this keeps the contract if a
            # row ever arrives without it (e.g. a NULL quality).
            results = [h for h in results if h.quality >= min_quality]
        for h in results:
            h.score *= _quality_mult(h.quality) * _recency_mult(h.created_ts)
        return results

    @staticmethod
    def _attribute(
        hits: list[Hit], lists: list[tuple[int, list[dict]]], n: int, k: int, min_quality: int
    ) -> None:
        """Set each hit's ``matched_queries``: the phrasings that would have returned it
        in their own top ``k``.

        Not "appeared in the candidate pool": every vector search returns its nearest
        rows whatever they are, so pool membership is true of nearly every hit and says
        nothing. Each phrasing is ranked with the same fusion and weighting as the
        combined search, so a single phrasing attributes every hit to itself.
        """
        if n == 1:
            for h in hits:
                h.matched_queries = [0]
            return
        top: list[set[str]] = []
        for qi in range(n):
            own = Store._fuse_many([lst for lst in lists if lst[0] == qi], min_quality)
            own.sort(key=lambda h: h.score, reverse=True)
            top.append({h.note_id for h in own[:k]})
        for h in hits:
            h.matched_queries = [qi for qi in range(n) if h.note_id in top[qi]]

    def _retrieve_all(
        self, tbl, vecs: list[list[float]], queries: list[str], pool: int, where: str | None
    ) -> list[tuple[int, list[dict]]]:
        lists: list[tuple[int, list[dict]]] = []
        for qi, (q, vec) in enumerate(zip(queries, vecs, strict=True)):
            vhits, fhits = self._retrieve(tbl, vec, q, pool, where)
            lists += [(qi, vhits), (qi, fhits)]
        return lists

    def search(
        self,
        query: str | list[str],
        *,
        k: int = 8,
        filters: dict | None = None,
        pool: int = 40,
        min_quality: int = 0,
    ) -> list[Hit]:
        """Hybrid search. Raises ValueError for a malformed filter, IndexDimMismatch
        when the index was built with another model, and RuntimeError when the vector
        search fails; a failed keyword search only adds to ``self.warnings``.

        ``query`` may be a list of phrasings: all are embedded in one batch, each runs
        its own vector and keyword search, and every list is fused by RRF into one
        ranking. Each hit's ``matched_queries`` says which phrasings found it.
        """
        self.warnings = []
        queries = normalize_queries(None, [query] if isinstance(query, str) else query)
        if not queries:
            return []
        where = self._where(filters)
        if min_quality > 0:
            # In the query, not after it: as a post-filter over a fixed 40-chunk pool,
            # low-quality stubs crowded out every qualifying note.
            qc = f"quality >= {int(min_quality)}"
            where = f"({where}) AND {qc}" if where else qc

        cached = self._tbl is not None
        tbl = self._existing_table()
        if tbl is None:
            return []
        self._check_dim(tbl)
        emb = _embed.get_embedder()
        # One phrasing keeps the single-query path; several are embedded in one batch.
        vecs = (
            [emb.embed_query(queries[0])] if len(queries) == 1 else emb.embed(queries, kind="query")
        )
        try:
            lists = self._retrieve_all(tbl, vecs, queries, pool, where)
        except IndexDimMismatch:
            raise
        except Exception:
            if not cached:
                raise
            # A long-lived Store's handle can outlive a drop by another process: reopen once.
            self._tbl, self.warnings = None, []
            tbl = self._existing_table()
            if tbl is None:
                return []
            lists = self._retrieve_all(tbl, vecs, queries, pool, where)

        results = self._fuse_many(lists, min_quality)
        if len(results) < k and pool < _MAX_POOL and any(len(h) >= pool for _, h in lists):
            # A full pool that collapsed to fewer than k notes: a few notes' chunks
            # crowded it. Widen once, reusing the query vectors.
            wide = min(pool * 3, _MAX_POOL)
            lists = self._retrieve_all(tbl, vecs, queries, wide, where)
            results = self._fuse_many(lists, min_quality)
        self.warnings = list(dict.fromkeys(self.warnings))

        ranked = sorted(results, key=lambda h: h.score, reverse=True)[:k]
        self._attribute(ranked, lists, len(queries), k, min_quality)
        return ranked
