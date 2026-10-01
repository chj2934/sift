"""Orchestration: vault notes -> chunks -> embeddings -> LanceDB, plus the
high-level search entrypoint used by both the CLI and the MCP server.

Indexing model
--------------
The index is keyed by FILE. Every chunk row carries the path it came from and the
file's mtime (taken before the file was read), and chunk ids are unique per file, so
two files sharing a frontmatter id (KEV and NVD twins of one CVE) are both indexed
and never clobber each other.

* `reindex` (CLI) and `sync` (long-lived processes) stat-walk the vault with the one
  enumeration rule (`walk_note_entries`), compare each file's mtime with the index,
  and parse and embed only new or changed files. Each flush is ONE atomic commit
  (`Store.replace_notes`, scoped by path). Rows of files that are gone are reaped,
  rows of notes whose body emptied are cleared, and rows of files that failed to
  load are kept (a half-typed YAML block must not drop a note from search).
* `index_notes` / `index_note` (ingest, MCP writes) replace the given files' rows
  in one commit, and clear a note whose body no longer yields chunks.
* Chunks are sized to the embedder's real window (`_chunk`), once per note; the
  chunker version and embed model are recorded next to the index so a stale index
  is reported instead of silently kept.

Nothing here prints: this module is reachable from the MCP server, whose stdout is
the JSON-RPC wire. Diagnostics go to the ``sift.pipeline`` logger.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sift.config import get_settings
from sift.index import embed as _embed
from sift.index.graph import build_link_index, expand_records
from sift.index.rerank import get_reranker
from sift.index.store import ChunkRow, Hit, Store, norm_path
from sift.quality import is_user_authored, score_note
from sift.vault.chunk import CHUNKER_VERSION, DEFAULT_SPECIAL_TOKENS, Chunk, chunk_markdown
from sift.vault.notes import (
    Note,
    load_note,
    report_unreadable,
    walk_note_entries,
    write_text_atomic,
)

if TYPE_CHECKING:
    from sift.vault.catalog import CatalogRow, VaultCatalog

log = logging.getLogger(__name__)

# Serialises this process's index writes (a background `sync` and an MCP `remember`
# can run on different threads). Never held while embedding.
_INDEX_LOCK = threading.RLock()

# A routine incremental pass refuses to reap more missing files than this floor or
# a fifth of the indexed files, whichever is larger: an unmounted drive or a wrong
# SIFT_VAULT_PATH looks exactly like "every note was deleted".
_REAP_FLOOR = 50

# Candidate chunks fetched per retriever. Window-sized chunking gives long notes more
# chunks, so a 40-row pool covered fewer distinct notes.
_MIN_POOL = 80
_MAX_POOL = 400

_INDEX_META_FILE = "index_meta.json"


# --------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def _created_ts(note: Note) -> float:
    """The note's date for the recency weight, epoch seconds (0 = unknown, neutral).

    `remember` and `capture_idea` notes carry no `created` date, which ranked the
    user's newest notes as if undated. For user-authored notes only, fall back to the
    capture time (`extra.captured`, which `resolve_idea` does not re-stamp) and then
    `ingested`. Undated bulk or distilled notes stay neutral.
    """
    m = note.meta
    d = m.created
    if not d and is_user_authored(m):
        extra = m.extra if isinstance(m.extra, dict) else {}
        d = _as_date(extra.get("captured")) or _as_date(m.ingested)
    if not d:
        return 0.0
    return datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()


def _passage_header(title: str, heading: str) -> str:
    """What is embedded in front of a chunk's text (after the model's passage prefix)."""
    return f"{title}\n{heading}\n" if heading else f"{title}\n"


def _passage(title: str, heading: str, text: str) -> str:
    """Text we embed for a chunk - prefixed with the note's identity."""
    return _passage_header(title, heading) + text


def _chunk(note: Note, emb: Any = None) -> list[Chunk]:
    """The note's chunks, sized to the embedder's window.

    The one place chunk_markdown is called, so every indexing path agrees. The context
    is exactly what gets embedded before each chunk (the model's passage prefix plus
    `_passage_header`), so ``special tokens + context + text`` fits ``max_seq_length``.
    Embedders without the token API (test fakes, older backends) fall back to
    chunk_markdown's conservative estimate and the default 512-token window.
    """
    body = note.body or ""
    if not body.strip():
        return []
    if emb is None:
        emb = _embed.get_embedder()
    title = note.meta.title or ""
    prefix = str(getattr(emb, "passage_prefix", "") or "")
    window = getattr(emb, "max_seq_length", None)
    count = getattr(emb, "count_tokens", None)
    special = getattr(emb, "num_special_tokens", None)
    return chunk_markdown(
        body,
        max_tokens=window if isinstance(window, int) and window > 0 else None,
        count_tokens=count if callable(count) else None,
        special_tokens=special
        if isinstance(special, int) and special >= 0
        else DEFAULT_SPECIAL_TOKENS,
        context=lambda heading: prefix + _passage_header(title, heading),
    )


def _stat_mtime(path: Path | None) -> float:
    if path is None:
        return 0.0
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _rows_for_note(
    note: Note,
    *,
    chunks: list[Chunk] | None = None,
    vectors: Sequence[Sequence[float]] | None = None,
    emb: Any = None,
) -> list[ChunkRow]:
    """Index rows for one note, built from the chunks that were embedded.

    Pass the chunk list the vectors came from: chunking twice and zipping the result
    against the vectors could silently misalign them, so the zip is strict. The stored
    mtime is the one taken before the file was read (`Note.mtime`), so an edit saved
    while the note was being embedded leaves the index older than the file and the
    next pass picks it up.
    """
    if chunks is None:
        chunks = _chunk(note, emb)
    if not chunks:
        return []
    m = note.meta
    if vectors is None:
        emb = emb if emb is not None else _embed.get_embedder()
        vectors = emb.embed([_passage(m.title, c.heading, c.text) for c in chunks], kind="passage")
    mtime = note.mtime if note.mtime is not None else _stat_mtime(note.path)
    quality = score_note(m, note.body)
    created_ts = _created_ts(note)
    rows: list[ChunkRow] = []
    for c, vec in zip(chunks, vectors, strict=True):
        rows.append(
            ChunkRow(
                note_id=m.id,
                slug=note.slug,
                type=m.type,
                title=m.title,
                heading=c.heading,
                text=c.text,
                chunk_index=c.index,
                vector=list(vec),
                source=m.source or "",
                url=m.url or "",
                cwe=m.cwe,
                tags=m.tags,
                severity=m.severity or "",
                program=m.program or "",
                path=str(note.path) if note.path else "",
                mtime=mtime,
                quality=quality,
                created_ts=created_ts,
            )
        )
    return rows


def _embed_rows(batch: Sequence[tuple[Note, list[Chunk]]], emb: Any) -> list[list[ChunkRow]]:
    """Rows for each (note, chunks) pair, from ONE embedding call over the batch."""
    passages = [_passage(n.meta.title, c.heading, c.text) for n, chunks in batch for c in chunks]
    vectors = emb.embed(passages, kind="passage") if passages else []
    if len(vectors) != len(passages):
        raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(passages)} passages")
    out: list[list[ChunkRow]] = []
    i = 0
    for n, chunks in batch:
        take = vectors[i : i + len(chunks)]
        i += len(chunks)
        out.append(_rows_for_note(n, chunks=chunks, vectors=take))
    return out


# --------------------------------------------------------------------------------
# Index metadata: which chunker and model built the index
# --------------------------------------------------------------------------------

_STALE_WARNED: set[tuple[str, str]] = set()
_STALE_WARNED_LOCK = threading.Lock()


def _model_name(emb: Any) -> str:
    return str(getattr(emb, "model_name", "") or get_settings().embed_model)


def index_meta(store: Store | None = None) -> dict | None:
    """What the index records about how it was built (chunker version, embed model),
    or None when nothing is recorded (an index built before this was tracked)."""
    store = store or Store()
    try:
        data = json.loads((store.db_path / _INDEX_META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _stamp_index(store: Store, model: str) -> None:
    data = {
        "chunker_version": CHUNKER_VERSION,
        "embed_model": model,
        "embed_dim": store.dim,
        "stamped": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    try:
        write_text_atomic(store.db_path / _INDEX_META_FILE, json.dumps(data, indent=2) + "\n")
    except Exception as exc:  # noqa: BLE001 - a missing record only costs a warning later
        log.warning("could not record the index metadata (%s)", exc)


def stale_index_reason(store: Store | None = None, model: str | None = None) -> str | None:
    """Why existing rows may have been chunked or embedded differently from what this
    code would do now, or None. An incremental reindex never re-chunks unchanged
    notes, so a mismatch needs one `sift reindex --force`."""
    store = store or Store()
    model = model or _model_name(_embed.get_embedder())
    meta = index_meta(store)
    if meta is None:
        return f"the index records no chunker version (built before {CHUNKER_VERSION})"
    if meta.get("chunker_version") != CHUNKER_VERSION:
        return (
            f"the index was chunked by {meta.get('chunker_version')!r}; "
            f"the current chunker is {CHUNKER_VERSION!r}"
        )
    if meta.get("embed_model") != model:
        return f"the index was embedded with {meta.get('embed_model')!r}; the current model is {model!r}"
    return None


def _warn_stale_once(store: Store, reason: str) -> None:
    key = (norm_path(store.db_path), reason)
    with _STALE_WARNED_LOCK:
        if key in _STALE_WARNED:
            return
        _STALE_WARNED.add(key)
    log.warning(
        "%s. An incremental reindex only re-chunks notes that changed; run "
        "`sift reindex --force` once.",
        reason,
    )


# --------------------------------------------------------------------------------
# Indexing given notes (ingest, MCP writes)
# --------------------------------------------------------------------------------


def _stale_index_paths(store: Store, note_ids: Iterable[str], paths: Iterable[str]) -> list[str]:
    """Stored paths of these notes' rows that replacing them must also clear.

    That is another spelling of a file being written (Windows case, a relative vault
    path), or a file that is no longer on disk: a renamed or moved note, or a twin that
    was merged away. A twin that still exists keeps its rows. Never raises: on a
    failure the write is scoped to the given paths and the next reindex reaps the rest.
    """
    ids = list(dict.fromkeys(i for i in note_ids if i))
    if not ids:
        return []
    exact = set(paths)
    mine = {norm_path(p) for p in exact}
    try:
        stored = store.stored_paths(ids)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not look up the indexed paths of %d note(s) (%s)", len(ids), exc)
        return []
    out: list[str] = []
    for p in stored:
        if not p or p in exact:
            continue
        if norm_path(p) in mine or not os.path.exists(p):
            out.append(p)
    return out


def index_notes(notes: Iterable[Note], store: Store | None = None, *, replace: bool = True) -> int:
    """Embed and index notes: one embedding call and ONE commit for the whole batch.

    With ``replace`` (the default) the notes' files end up with exactly these rows: a
    note whose body no longer yields chunks has its rows cleared, and rows left under
    an old path of the note (a rename) are cleared too. Rows are scoped by file, so
    re-indexing one of two files that share an id leaves the other's rows alone; a
    note that was never saved (no path) is scoped by its id. Returns rows written.
    """
    notes = list(notes)
    if not notes:
        return 0
    store = store or Store()
    emb = _embed.get_embedder() if any((n.body or "").strip() for n in notes) else None
    batch = [(n, _chunk(n, emb)) for n in notes]
    per_note = _embed_rows(batch, emb)
    rows = [r for nrows in per_note for r in nrows]

    with _INDEX_LOCK:
        was_empty = store.count() == 0
        if replace:
            pathless_ids = [n.meta.id for n in notes if not n.path]
            paths = [str(n.path) for n in notes if n.path]
            stale = (
                []
                if was_empty
                else _stale_index_paths(store, [n.meta.id for n in notes if n.path], paths)
            )
            store.replace_notes(rows, note_ids=pathless_ids, paths=paths + stale)
        else:
            store.add_chunks(rows)
        # Record the chunker only for a table this call started. Count again: `count()`
        # answers 0 on a read error, and an old index must not be stamped as current.
        if was_empty and rows and store.count() == len(rows):
            _stamp_index(store, _model_name(emb))
    _note_synced(notes)
    return len(rows)


def index_note(note: Note, store: Store | None = None) -> int:
    """`index_notes` for one note: one commit, and an emptied body clears its rows."""
    return index_notes([note], store)


# --------------------------------------------------------------------------------
# Reindex: the vault -> the index
# --------------------------------------------------------------------------------


@dataclass
class ReindexStats:
    notes: int = 0  # files (re)embedded this pass
    chunks: int = 0
    # Files whose body yields no chunks (rows they had were cleared).
    skipped: int = 0
    # Files whose mtime matched the index, so they were not re-embedded.
    unchanged: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    # Files whose index rows were removed: gone from disk, or emptied.
    removed: int = 0
    # Files that failed to load (empty, no frontmatter, invalid). Their existing rows
    # are kept; they are reported once through logging.
    unreadable: int = 0
    # Files gone from disk whose rows were KEPT because the reap looked like a mass
    # delete (unmounted drive, wrong vault path). `reindex(allow_mass_reap=True)` or
    # `--force` removes them.
    reap_refused: int = 0
    # Files edited while they were being embedded: not written, picked up next pass.
    requeued: int = 0
    # Ids carried by more than one file. Both files are indexed; report only.
    duplicate_ids: int = 0
    # `sync` only: changed files left for `sift reindex` (over its max_notes).
    deferred: int = 0
    walked: int = 0  # files the walk found: the progress total
    # Why existing rows may predate the current chunker or model (see stale_index_reason).
    stale_index: str | None = None
    # Store.optimize() report, when maintenance ran.
    maintenance: dict | None = None


def _vault_dir(vault: Path | str | None) -> Path:
    return Path(os.path.abspath(vault if vault is not None else get_settings().resolved_vault()))


def count_notes(vault: Path | None = None) -> int:
    """How many files `reindex` will walk: the progress-bar total. A stat walk with the
    same enumeration rule as `iter_notes` and the catalog; nothing is parsed."""
    return sum(1 for _ in walk_note_entries(_vault_dir(vault)))


def _changed_since_read(note: Note) -> bool:
    """True if the file no longer has the mtime it had when the note was read."""
    if note.path is None or note.mtime is None:
        return False
    try:
        return os.stat(note.path).st_mtime != note.mtime
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _reindex(
    vault: Path,
    store: Store,
    *,
    entries: list[tuple[Path, os.stat_result]] | None,
    force: bool,
    batch: int,
    on_progress: Callable[[int], None] | None,
    allow_mass_reap: bool,
    maintain: bool,
    max_notes: int | None = None,
) -> ReindexStats:
    stats = ReindexStats()

    # 1. Snapshot the index BEFORE walking, so a note written and indexed during the
    #    walk is never mistaken for an orphan.
    if force:
        with _INDEX_LOCK:
            store.drop()
        snapshot: list[tuple[str, str, float]] = []
    else:
        snapshot = store.indexed_files()
        if not snapshot and store.count() > 0:
            raise RuntimeError(
                "could not read which notes are indexed; refusing to re-embed the whole "
                "vault. Retry, or run `sift reindex --force`."
            )
    by_file: dict[str, list[tuple[str, str, float]]] = {}
    for nid, p, m in snapshot:
        if p:  # rows without a path cannot be matched to a file; they are left alone
            by_file.setdefault(norm_path(p), []).append((nid, p, m))

    if entries is None:
        entries = list(walk_note_entries(vault)) if vault.is_dir() else []
    stats.walked = len(entries)

    # 2. Plan from stats alone: unchanged files are never parsed.
    ids_seen: dict[str, int] = {}
    walked_files: set[str] = set()
    todo: list[Path] = []
    for path, st in entries:
        key = norm_path(path)
        walked_files.add(key)
        held = by_file.get(key)
        # Float equality is right here: both sides are the same stat value
        # round-tripped through the store, not a computed quantity.
        if held and all(m == st.st_mtime for _n, _p, m in held):
            stats.unchanged += 1
            for nid in {n for n, _p, _m in held}:
                ids_seen[nid] = ids_seen.get(nid, 0) + 1
            continue
        todo.append(path)

    # Rows of files the walk did not find and that are gone from disk. A file that
    # still exists but was not walked (an ignored folder, another vault) keeps its rows.
    missing = list(
        dict.fromkeys(
            p
            for key, held in by_file.items()
            if key not in walked_files
            for _n, p, _m in held
            if not os.path.exists(p)
        )
    )
    missing_files = {norm_path(p) for p in missing}
    if missing and not allow_mass_reap:
        floor = max(_REAP_FLOOR, len(by_file) // 5)
        if not entries or len(missing_files) > floor:
            stats.reap_refused = len(missing_files)
            log.warning(
                "%d indexed notes are gone from %s (%d files walked); keeping their index "
                "rows in case the vault is unmounted or misconfigured. If they really were "
                "deleted, run `sift reindex --force`.",
                len(missing_files),
                vault,
                len(entries),
            )
            missing, missing_files = [], set()

    if max_notes is not None and len(todo) > max_notes:
        stats.deferred = len(todo)
        log.warning(
            "%d notes changed since the index was built; leaving them for `sift reindex` "
            "(a background sync only handles small changes).",
            len(todo),
        )
        return stats

    emb: Any = None
    model = ""
    if todo or (not force and snapshot):
        emb = _embed.get_embedder()
        model = _model_name(emb)
    if force or (not snapshot and todo):
        _stamp_index(store, model or _model_name(_embed.get_embedder()))
    elif snapshot:
        stats.stale_index = stale_index_reason(store, model)
        if stats.stale_index:
            _warn_stale_once(store, stats.stale_index)

    processed = stats.unchanged
    if on_progress and processed:
        on_progress(processed)

    # 3. Load, chunk, embed and write what changed.
    buf: list[tuple[Note, list[Chunk]]] = []
    buf_chunks = 0
    clear: list[str] = []  # stored paths whose rows go (emptied notes, vanished files)
    last_report = 0

    def stored_spellings(path: Path | None) -> list[str]:
        if path is None:
            return []
        return [p for _n, p, _m in by_file.get(norm_path(path), ())]

    def flush(*, final: bool) -> None:
        nonlocal buf, buf_chunks, clear
        per_note = _embed_rows(buf, emb) if buf else []
        with _INDEX_LOCK:
            rows: list[ChunkRow] = []
            scope: list[str] = []
            for (note, chunks), nrows in zip(buf, per_note, strict=True):
                if _changed_since_read(note):
                    # Edited while it was embedded: writing now would store old text
                    # under a stale mtime. Its old rows stay; the next pass redoes it.
                    stats.requeued += 1
                    stats.notes -= 1
                    stats.chunks -= len(chunks)
                    stats.by_type[note.meta.type] -= 1
                    continue
                rows.extend(nrows)
                scope.append(str(note.path))
                scope.extend(stored_spellings(note.path))
            scope.extend(clear)
            if final:
                scope.extend(p for p in missing if not os.path.exists(p))
            if force:
                store.add_chunks(rows)
            elif rows or scope:
                store.replace_notes(rows, note_ids=[], paths=scope)
        buf, buf_chunks, clear = [], 0, []

    for path in todo:
        processed += 1
        try:
            note = load_note(path)
        except FileNotFoundError:
            # Removed between the walk and the read.
            gone = stored_spellings(path)
            if gone:
                clear.extend(gone)
                stats.removed += 1
            continue
        except Exception as exc:  # noqa: BLE001 - one bad file never stops a reindex
            report_unreadable(path, exc, vault=vault)
            stats.unreadable += 1  # its existing rows (if any) are kept
            continue
        ids_seen[note.meta.id] = ids_seen.get(note.meta.id, 0) + 1
        chunks = _chunk(note, emb)
        if not chunks:
            stats.skipped += 1
            had = stored_spellings(path)
            if had:
                clear.extend(had)
                stats.removed += 1
            continue
        buf.append((note, chunks))
        buf_chunks += len(chunks)
        stats.notes += 1
        stats.chunks += len(chunks)
        stats.by_type[note.meta.type] = stats.by_type.get(note.meta.type, 0) + 1
        if buf_chunks >= batch:
            flush(final=False)
            if on_progress:
                on_progress(processed)
            elif stats.notes - last_report >= 2000:
                last_report = stats.notes
                log.info("  .. %d notes indexed", stats.notes)

    flush(final=True)
    stats.removed += len(missing_files)
    if on_progress:
        on_progress(stats.walked)

    stats.duplicate_ids = sum(1 for c in ids_seen.values() if c > 1)
    if stats.duplicate_ids:
        log.info(
            "%d note ids are carried by more than one file; every file is indexed. "
            "`sift doctor` lists them.",
            stats.duplicate_ids,
        )
    if maintain:
        stats.maintenance = store.optimize()
    return stats


def reindex(
    vault: Path | None = None,
    *,
    force: bool = False,
    batch: int = 512,
    on_progress: Callable[[int], None] | None = None,
    allow_mass_reap: bool = False,
    maintain: bool = True,
) -> ReindexStats:
    """Bring the index in line with the vault, embedding chunks in large batches.

    Incremental by default: files whose mtime matches the index are not even parsed.
    New and changed files are re-embedded, rows of deleted files and emptied notes are
    removed, and rows of files that fail to load are kept. Refuses to reap a mass
    delete unless ``allow_mass_reap``. ``force`` drops the table and rebuilds it.
    ``maintain`` runs `Store.optimize()` at the end (routine compaction, FTS index
    creation after a drop); CLI only, never from an MCP tool.

    `on_progress` receives the number of files handled so far; the last call is the
    walk total (`ReindexStats.walked`, which `count_notes` predicts).
    """
    return _reindex(
        _vault_dir(vault),
        Store(),
        entries=None,
        force=force,
        batch=batch,
        on_progress=on_progress,
        allow_mass_reap=allow_mass_reap,
        maintain=maintain,
    )


# --------------------------------------------------------------------------------
# Sync: a cheap incremental reindex for long-lived processes (the MCP server)
# --------------------------------------------------------------------------------

SYNC_MIN_INTERVAL = 5.0
SYNC_MAX_NOTES = 200


@dataclass
class _SyncState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    # norm path -> (st_mtime_ns, st_size) as of the last completed sync
    files: dict[str, tuple[int, int]] | None = None
    checked_at: float | None = None


_SYNC_STATES: dict[str, _SyncState] = {}
_SYNC_STATES_LOCK = threading.Lock()


def _sync_state(vault: Path) -> _SyncState:
    key = norm_path(vault)
    with _SYNC_STATES_LOCK:
        state = _SYNC_STATES.get(key)
        if state is None:
            state = _SYNC_STATES[key] = _SyncState()
        return state


def _note_synced(notes: Iterable[Note]) -> None:
    """Tell running syncs about files this process just indexed, so its own writes
    (remember, capture_idea, resolve_idea) do not trigger an index rescan."""
    with _SYNC_STATES_LOCK:
        states = [(k, s) for k, s in _SYNC_STATES.items() if s.files is not None]
        if not states:
            return
        for note in notes:
            if note.path is None:
                continue
            key = norm_path(note.path)
            for vkey, state in states:
                if key.startswith(vkey + os.sep) and state.files is not None:
                    try:
                        st = os.stat(note.path)
                    except OSError:
                        state.files.pop(key, None)
                    else:
                        state.files[key] = (st.st_mtime_ns, st.st_size)


def sync(
    vault: Path | None = None,
    *,
    min_interval: float = SYNC_MIN_INTERVAL,
    max_notes: int | None = SYNC_MAX_NOTES,
    wait: bool = False,
) -> ReindexStats | None:
    """Pick up vault edits made outside sift (Obsidian) without a CLI reindex.

    Cheap when nothing changed: one stat walk, compared with the walk of the last
    sync, and no index access at all. When something changed it runs the incremental
    reindex, minus maintenance (no `optimize`, no FTS rebuild): new and edited files
    are re-embedded and deleted ones reaped. More than ``max_notes`` changed files are
    left for `sift reindex` - a background sync must not embed a whole vault.

    Returns None when it did nothing: called again within ``min_interval`` seconds,
    another sync is running (unless ``wait``), nothing changed, or it failed (logged).
    Never raises.
    """
    try:
        vault = _vault_dir(vault)
        state = _sync_state(vault)
    except Exception as exc:  # noqa: BLE001
        log.warning("index sync skipped (%s)", exc)
        return None
    if not state.lock.acquire(blocking=wait):
        return None
    try:
        now = time.monotonic()
        if (
            state.checked_at is not None
            and min_interval > 0
            and now - state.checked_at < min_interval
        ):
            return None
        state.checked_at = now
        if not vault.is_dir():
            return None
        entries = list(walk_note_entries(vault))
        files = {norm_path(p): (st.st_mtime_ns, st.st_size) for p, st in entries}
        with _SYNC_STATES_LOCK:
            if state.files == files:
                return None
        try:
            stats = _reindex(
                vault,
                Store(),
                entries=entries,
                force=False,
                batch=512,
                on_progress=None,
                allow_mass_reap=False,
                maintain=False,
                max_notes=max_notes,
            )
        except Exception as exc:  # noqa: BLE001 - a background sync must never raise
            log.warning(
                "index sync failed (%s: %s); the next sync retries", type(exc).__name__, exc
            )
            return None
        with _SYNC_STATES_LOCK:
            state.files = files
        if stats.notes or stats.removed:
            log.info("index sync: %d notes re-embedded, %d removed", stats.notes, stats.removed)
        return stats
    finally:
        state.lock.release()


def _sync_quietly(vault: Path, min_interval: float) -> None:
    try:
        sync(vault, min_interval=min_interval)
    except BaseException as exc:  # noqa: BLE001 - never let a daemon thread die loudly
        log.warning("index sync thread failed (%s)", exc)


def sync_in_background(
    vault: Path | None = None, *, min_interval: float = SYNC_MIN_INTERVAL
) -> threading.Thread | None:
    """Start `sync` on a daemon thread, unless one is running or ran within
    ``min_interval`` seconds. Returns the thread, or None if none was started.

    For the MCP server: call it at startup and at the top of search_memory, so a tool
    call never waits for an embedding. Never raises, never prints."""
    try:
        vault = _vault_dir(vault)
        state = _sync_state(vault)
        if state.lock.locked():
            return None
        if state.checked_at is not None and time.monotonic() - state.checked_at < min_interval:
            return None
        t = threading.Thread(
            target=_sync_quietly, args=(vault, min_interval), name="sift-index-sync", daemon=True
        )
        t.start()
        return t
    except Exception as exc:  # noqa: BLE001
        log.warning("could not start the index sync (%s)", exc)
        return None


# --------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------


@dataclass
class SearchResult:
    hits: list[Hit]
    linked: list[dict]  # expanded neighbour notes: {slug, note_id, title, type, path, url}
    # Why the result may be partial (e.g. keyword search unavailable). Empty when clean.
    warnings: list[str] = field(default_factory=list)


def _linked(hits: list[Hit], warnings: list[str]) -> list[dict]:
    """1-hop neighbours of the hits, seeded by note id (exact, unlike a slug)."""
    try:
        link_index = build_link_index(get_settings().resolved_vault())
        recs = expand_records(
            list(dict.fromkeys(h.note_id for h in hits)), link_index, hops=1, limit=12
        )
    except Exception as exc:  # noqa: BLE001 - the hits are still good without neighbours
        log.warning("link expansion failed (%s)", exc)
        warnings.append(f"link expansion unavailable: {exc}")
        return []
    out: list[dict] = []
    for rec in recs:
        meta = rec.meta
        path = getattr(rec, "path", None)
        out.append(
            {
                "slug": rec.slug,
                "note_id": getattr(rec, "note_id", None) or meta.id,
                "title": meta.title,
                "type": meta.type,
                "path": str(path) if path else "",
                "url": meta.url or "",
            }
        )
    return out


def search(
    query: str,
    *,
    k: int = 8,
    filters: dict | None = None,
    expand_links: bool = False,
    min_quality: int = 0,
) -> SearchResult:
    """Hybrid search. Raises ValueError for a malformed filter, IndexDimMismatch (a
    RuntimeError) when the index was built with another model, and RuntimeError when
    vector search fails; a degraded keyword search is reported in `warnings`."""
    store = Store()
    reranker = get_reranker()
    want = max(k, 20) if reranker else k
    pool = min(_MAX_POOL, max(_MIN_POOL, want * 10))
    hits = store.search(query, k=want, filters=filters, pool=pool, min_quality=min_quality)
    warnings = list(store.warnings)
    hits = reranker.rerank(query, hits, top_k=k) if reranker else hits[:k]

    linked = _linked(hits, warnings) if expand_links and hits else []
    return SearchResult(hits=hits, linked=linked, warnings=list(dict.fromkeys(warnings)))


# --------------------------------------------------------------------------------
# Resolving a note reference (get_note, resolve_idea)
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class NoteLookup:
    """What a note reference resolved to.

    `note` is set when exactly one note answers to the key (loaded fresh from disk, so
    callers may mutate and save it). When several do, `note` is None and `candidates`
    lists them: a reader should show them, a writer must refuse rather than guess.
    """

    note: Note | None = None
    candidates: tuple[CatalogRow, ...] = ()

    @property
    def ambiguous(self) -> bool:
        return self.note is None and len(self.candidates) > 1

    def candidate_info(self) -> list[dict]:
        return [
            {
                "note_id": r.id,
                "slug": r.slug,
                "title": r.title,
                "type": r.type,
                "path": str(r.path),
            }
            for r in self.candidates
        ]


def _looks_like_path(key: str) -> bool:
    return key.lower().endswith(".md") or "/" in key or "\\" in key


def _catalog_rows(cat: VaultCatalog, key: str) -> tuple[CatalogRow, ...]:
    rows = cat.by_id(key)  # an exact id always wins, even over a path-looking key
    if rows:
        return rows
    if _looks_like_path(key):
        row = cat.by_path(key)  # only catalogued vault notes: never an arbitrary file
        if row is not None:
            return (row,)
    return cat.lookup(key)


def resolve_note(key: str, *, vault: Path | None = None, max_age: float = 1.0) -> NoteLookup:
    """Resolve a note id, slug, legacy 80-char slug, filename or vault path.

    Strongest match first, through the catalog: exact id > slug > legacy slug >
    filename stem; a path is accepted only if it is a catalogued note of this vault.
    The note is loaded from disk and checked against the catalog row, so a stale row
    (an id edited a moment ago) is re-read instead of served.
    """
    from sift.vault.catalog import fresh_catalog

    key = str(key or "").strip()
    if not key:
        return NoteLookup()
    cat = fresh_catalog(_vault_dir(vault), max_age=max_age)
    for attempt in range(2):
        rows = _catalog_rows(cat, key)
        if len(rows) > 1:
            return NoteLookup(candidates=rows)
        if rows:
            row = rows[0]
            try:
                note = load_note(row.path)
            except Exception:  # noqa: BLE001 - gone or now unreadable: re-walk once
                note = None
            if note is not None and note.meta.id == row.id:
                return NoteLookup(note=note)
            cat.invalidate([row.path])
        if attempt == 0:
            cat.ensure_fresh(0.0)  # a miss or a stale row: one fresh walk, then answer
    return NoteLookup()


def get_note_by_slug(slug: str) -> Note | None:
    """The one note `slug` (or an id, filename or vault path) names, or None - also
    when several notes answer to it; use `resolve_note` to see the candidates."""
    return resolve_note(slug).note
