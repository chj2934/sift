"""A persisted catalog of the vault: id / slug / filename -> file, plus listing rows.

Answers "which file carries id X?", "which note is slug or filename Y?" and "every
note's metadata" without parsing 13.8k files per call; `list_notes`, `stats`,
`get_note` and `resolve_idea` used to pay 2-5 s for each of those. A row is one
note file's frontmatter summary plus its outgoing links, keyed by path and
validated by ``(st_mtime_ns, st_size)``.

Refresh is incremental. A stat walk - the same enumeration rule as `iter_notes`,
`notes.walk_note_entries` - re-parses only new or changed files and drops vanished
ones. Two extra rules keep a stat-validated cache honest:

* **Racy files are re-read.** A file modified within two seconds of being read is
  read again on the next walk (git's racy-clean rule): Windows keeps the same
  mtime across quick same-size rewrites often enough to matter.
* **Folder mtimes.** The walk records each folder's mtime. `ensure_fresh(max_age)`
  skips the walk only while the catalog is younger than `max_age` *and* no folder
  changed, so a note created or renamed by Obsidian or another sift process is seen
  at once, while a bulk ingest does not re-walk the vault per saved note (sift's own
  writes update the catalog directly, see `note_saved`).

The catalog is advisory. It is cached as JSON in sift's state dir (``<db>/_sift``,
never inside the vault); a cache file that is missing, corrupt, from another
version or for another vault is ignored, every cached row is re-validated by the
first walk, and every lookup a write depends on is verified against the file on
disk (`notes._find_carriers`). Rows are immutable plain data, never a `Note` a
caller could mutate; load the note from `row.path` for its body.

Thread-safe: FastMCP runs sync tools in a threadpool. Nothing here prints; an
unreadable file is reported once through `notes.report_unreadable` (logging).

The first build of the real vault parses every note once (seconds); later processes
start from the cached file and pay one stat walk. A long-lived server can warm it
off the request path with ``threading.Thread(target=get_catalog(v).ensure_fresh)``.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slugify import slugify

from sift.vault.notes import (
    LEGACY_SLUG_MAX,
    Note,
    describe_error,
    load_note,
    report_unreadable,
    state_dir,
    vault_key,
    walk_note_entries,
    write_bytes_atomic,
)

log = logging.getLogger(__name__)

CATALOG_VERSION = 1
# A file modified this close to when it was read is read again on the next walk.
_RACY_NS = 2_000_000_000
# Dirty catalogs are written back at most this often (and once at exit): a bulk
# ingest changes the catalog on every note, and the real vault's file is megabytes.
_PERSIST_EVERY_S = 30.0


def _key(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _order(key: str) -> list[str]:
    """Sort key giving `iter_notes` order (part-wise, so `a/b.md` < `a.md`)."""
    return key.split(os.sep)


def _cap(slug: str) -> str:
    """``slugify(text, max_length=80)`` given ``slugify(text)`` (python-slugify cuts
    and strips the separator, nothing more)."""
    return slug[:LEGACY_SLUG_MAX].strip("-")


@dataclass(frozen=True, slots=True)
class CatalogRow:
    """One note file, as the catalog knows it."""

    path: Path
    rel: str  # vault-relative, '/'-separated
    mtime_ns: int
    size: int
    id: str
    type: str
    title: str
    slug: str  # Note.slug: the whole id, slugified
    source: str | None = None
    url: str | None = None
    program: str | None = None
    severity: str | None = None
    status: str | None = None  # extra.status (captured ideas)
    created: str | None = None  # ISO date
    tags: tuple[str, ...] = ()
    links: tuple[str, ...] = ()  # meta.links + body [[wikilinks]], as slugs

    @property
    def legacy_slug(self) -> str:
        """The pre-uncapped slug (cut at 80): what old index rows and links carry."""
        return _cap(self.slug)

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def folder(self) -> str:
        """The top-level vault folder the file sits in ('' for the vault root)."""
        head, sep, _rest = self.rel.partition("/")
        return head if sep else ""


@dataclass(frozen=True, slots=True)
class _Entry:
    path: Path
    mtime_ns: int
    size: int
    seen_ns: int  # wall clock just before the file was read
    row: CatalogRow | None  # None: not a usable note
    error: str | None = None  # why not (one line, never file content)


def _fresh(entry: _Entry, st: os.stat_result) -> bool:
    return (
        entry.mtime_ns == st.st_mtime_ns
        and entry.size == st.st_size
        and st.st_mtime_ns + _RACY_NS < entry.seen_ns
    )


def _dir_mtime(path: Path | str) -> int:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return -1


class VaultCatalog:
    """The catalog of one vault. Get it with `get_catalog`; call `ensure_fresh`
    before a lookup that should reflect the disk."""

    def __init__(self, vault: Path, cache_file: Path | None) -> None:
        self.vault = Path(vault)
        self.cache_file = cache_file
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}
        self._by_id: dict[str, dict[str, CatalogRow]] = {}
        self._by_slug: dict[str, dict[str, CatalogRow]] = {}
        self._by_legacy: dict[str, dict[str, CatalogRow]] = {}
        self._by_stem: dict[str, dict[str, CatalogRow]] = {}
        self._sorted: list[CatalogRow] | None = None
        self._dirs: dict[str, tuple[str, int]] = {}
        self._walked_at: float | None = None
        self._generation = 0
        self._dirty = False
        self._persisted_at: float | None = None
        self._load()

    # ---- freshness ---------------------------------------------------------------
    @property
    def generation(self) -> int:
        """Bumped whenever any row changes; a cheap cache key for derived data."""
        return self._generation

    def ensure_fresh(self, max_age: float = 0.0) -> int:
        """Re-walk unless the last walk is younger than `max_age` seconds and no
        folder changed since. Returns the generation."""
        with self._lock:
            if (
                max_age > 0
                and self._walked_at is not None
                and time.monotonic() - self._walked_at < max_age
                and not self._dirs_changed()
            ):
                return self._generation
            return self.refresh()

    def refresh(self, *, force: bool = False) -> int:
        """Stat-walk the vault: re-parse new and changed files (every file with
        `force`), drop vanished ones. Returns the generation."""
        with self._lock:
            dirs: dict[str, tuple[str, int]] = {}

            def on_dir(d: Path) -> None:
                dirs[_key(d)] = (str(d), _dir_mtime(d))

            seen: set[str] = set()
            changed = False
            for path, st in walk_note_entries(self.vault, on_dir=on_dir):
                key = _key(path)
                old = self._entries.get(key)
                if not force and old is not None and _fresh(old, st):
                    seen.add(key)
                    continue
                new = self._parse(path, st, old)
                if new is None:
                    continue  # vanished mid-walk, or unreadable right now with no history
                seen.add(key)
                if old is None or old.row != new.row or old.error != new.error:
                    changed = True
                self._put(key, new)
            for key in [k for k in self._entries if k not in seen]:
                self._drop(key)
                changed = True
            self._dirs = dirs
            self._walked_at = time.monotonic()
            if changed:
                self._generation += 1
                self._dirty = True
            self._maybe_persist()
            return self._generation

    def invalidate(self, paths: Iterable[Path | str]) -> None:
        """Forget these files, so the next walk reads them again whatever their stat
        says (a row disagreed with its file)."""
        with self._lock:
            hit = False
            for p in paths:
                if _key(p) in self._entries:
                    self._drop(_key(p))
                    hit = True
            if hit:
                self._generation += 1
                self._walked_at = None

    def _dirs_changed(self) -> bool:
        return any(_dir_mtime(path) != mtime for path, mtime in self._dirs.values())

    def _parse(self, path: Path, st: os.stat_result, old: _Entry | None) -> _Entry | None:
        seen_ns = time.time_ns()  # before the read: racy-clean errs towards re-reading
        try:
            note = load_note(path)
        except FileNotFoundError:
            return None
        except OSError:
            return old  # locked by another program right now: keep what we had
        except Exception as exc:  # noqa: BLE001 - one bad file never stops a walk
            report_unreadable(path, exc, vault=self.vault)
            return _Entry(path, st.st_mtime_ns, st.st_size, seen_ns, None, describe_error(exc))
        row = self._row(note, path, st.st_mtime_ns, st.st_size)
        return _Entry(path, st.st_mtime_ns, st.st_size, seen_ns, row)

    def _row(self, note: Note, path: Path, mtime_ns: int, size: int) -> CatalogRow:
        meta = note.meta
        extra = meta.extra if isinstance(meta.extra, dict) else {}
        status = extra.get("status")
        try:
            rel = path.relative_to(self.vault).as_posix()
        except ValueError:
            rel = path.as_posix()
        return CatalogRow(
            path=path,
            rel=rel,
            mtime_ns=mtime_ns,
            size=size,
            id=meta.id,
            type=meta.type,
            title=meta.title,
            slug=note.slug,
            source=meta.source,
            url=meta.url,
            program=meta.program,
            severity=meta.severity,
            status=None if status is None else str(status),
            created=meta.created.isoformat() if meta.created else None,
            tags=tuple(meta.tags),
            links=tuple(note.all_links()),
        )

    # ---- index maintenance -----------------------------------------------------
    def _indexes(self, row: CatalogRow) -> tuple[tuple[dict[str, dict[str, CatalogRow]], str], ...]:
        return (
            (self._by_id, row.id),
            (self._by_slug, row.slug),
            (self._by_legacy, row.legacy_slug),
            (self._by_stem, row.path.stem.casefold()),
        )

    def _put(self, key: str, entry: _Entry) -> None:
        self._drop(key)
        self._entries[key] = entry
        if entry.row is not None:
            for index, k in self._indexes(entry.row):
                index.setdefault(k, {})[key] = entry.row
        self._sorted = None

    def _drop(self, key: str) -> None:
        old = self._entries.pop(key, None)
        if old is None:
            return
        if old.row is not None:
            for index, k in self._indexes(old.row):
                bucket = index.get(k)
                if bucket is not None:
                    bucket.pop(key, None)
                    if not bucket:
                        del index[k]
        self._sorted = None

    # ---- sift's own writes -----------------------------------------------------
    def note_saved(self, note: Note, *, previous: Path | None = None) -> None:
        """Record a note sift just wrote (and, on a rename, forget `previous`).

        The folders touched are re-stamped, so sift's own write does not look like
        an outside change and trigger a walk; the row is marked racy, so the next
        walk re-reads it anyway.
        """
        path = note.path
        if path is None:
            return
        with self._lock:
            if previous is not None and _key(previous) != _key(path):
                self._drop(_key(previous))
                self._restamp(Path(previous).parent)
            if not self._walkable(path):
                self._drop(_key(path))
            else:
                try:
                    st = path.stat()
                except OSError:
                    self._drop(_key(path))
                else:
                    row = self._row(note, path, st.st_mtime_ns, st.st_size)
                    entry = _Entry(path, st.st_mtime_ns, st.st_size, time.time_ns(), row)
                    self._put(_key(path), entry)
                self._restamp(path.parent)
            self._generation += 1
            self._dirty = True

    def note_removed(self, path: Path) -> None:
        """Forget a file sift just moved away (soft delete)."""
        with self._lock:
            self._drop(_key(path))
            self._restamp(Path(path).parent)
            self._generation += 1
            self._dirty = True

    def _restamp(self, folder: Path) -> None:
        key = _key(folder)
        if key in self._dirs:
            self._dirs[key] = (str(folder), _dir_mtime(folder))

    def _walkable(self, path: Path) -> bool:
        """Would the walk list this file? (Structural rules; configured ignores are
        corrected by the next walk.)"""
        try:
            rel = os.path.relpath(_key(path), _key(self.vault))
        except ValueError:  # another drive
            return False
        parts = Path(rel).parts
        if not parts or parts[0] == ".." or any(p.startswith((".", "_")) for p in parts):
            return False
        name = parts[-1]
        return name.lower().endswith(".md") and name.casefold() != "readme.md"

    # ---- lookups -----------------------------------------------------------------
    @staticmethod
    def _get(index: dict[str, dict[str, CatalogRow]], key: str) -> tuple[CatalogRow, ...]:
        bucket = index.get(key)
        if not bucket:
            return ()
        return tuple(bucket[k] for k in sorted(bucket, key=_order))

    def by_id(self, note_id: str) -> tuple[CatalogRow, ...]:
        """Every file carrying exactly this frontmatter id, in path order."""
        with self._lock:
            return self._get(self._by_id, str(note_id))

    def by_slug(self, slug: str) -> tuple[CatalogRow, ...]:
        """Notes whose (uncapped) `Note.slug` is `slug`."""
        with self._lock:
            return self._get(self._by_slug, slug)

    def by_legacy_slug(self, slug: str) -> tuple[CatalogRow, ...]:
        """Notes whose pre-uncapped 80-char slug is `slug` (several: ambiguous)."""
        with self._lock:
            return self._get(self._by_legacy, _cap(slug))

    def by_filename(self, name: str) -> tuple[CatalogRow, ...]:
        """Notes whose filename stem is `name` (case-insensitive, `.md` and a folder
        prefix optional): how Obsidian resolves ``[[Name]]``."""
        stem = str(name).replace("\\", "/").rsplit("/", 1)[-1].strip()
        if stem.lower().endswith(".md"):
            stem = stem[:-3]
        with self._lock:
            return self._get(self._by_stem, stem.casefold())

    def by_path(self, path: Path | str) -> CatalogRow | None:
        """The row for `path` (absolute, or relative to the vault) - only if the file
        is a catalogued note, which also confines a caller-supplied path to the
        vault's note files."""
        p = Path(path)
        if not p.is_absolute():
            p = self.vault / p
        with self._lock:
            entry = self._entries.get(_key(p))
            return entry.row if entry is not None else None

    def lookup(self, key: str) -> tuple[CatalogRow, ...]:
        """Resolve a note reference, strongest match first.

        Tiers: exact id > slug (uncapped) > legacy 80-char slug > filename stem. The
        first tier with any match is returned; more than one row means the reference
        is ambiguous and the caller must not guess (a write must refuse).
        """
        key = str(key or "").strip()
        if not key:
            return ()
        with self._lock:
            rows = self._get(self._by_id, key)
            if rows:
                return rows
            slug = slugify(key)
            if slug:
                rows = self._get(self._by_slug, slug) or self._get(self._by_legacy, _cap(slug))
                if rows:
                    return rows
        return self.by_filename(key)

    def rows(self, note_type: str | None = None) -> list[CatalogRow]:
        """Every note, in path order. With `note_type`, the notes under that top-level
        folder - the same scope as ``iter_notes(vault, note_type=...)``."""
        with self._lock:
            if self._sorted is None:
                self._sorted = [
                    e.row
                    for _k, e in sorted(self._entries.items(), key=lambda kv: _order(kv[0]))
                    if e.row is not None
                ]
            out = self._sorted
        if note_type:
            want = os.path.normcase(note_type)
            return [r for r in out if os.path.normcase(r.folder) == want]
        return list(out)

    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows():
            counts[row.type] = counts.get(row.type, 0) + 1
        return counts

    def skipped(self) -> list[tuple[Path, str]]:
        """Files that look like notes but are not usable (empty, no frontmatter,
        invalid), with a one-line reason - for `stats`, never file content."""
        with self._lock:
            items = sorted(self._entries.items(), key=lambda kv: _order(kv[0]))
            return [(e.path, e.error or "unreadable") for _k, e in items if e.row is None]

    def duplicate_ids(self) -> dict[str, tuple[Path, ...]]:
        """Ids carried by more than one file (report only; nothing is deleted)."""
        with self._lock:
            return {
                nid: tuple(bucket[k].path for k in sorted(bucket, key=_order))
                for nid, bucket in sorted(self._by_id.items())
                if len(bucket) > 1
            }

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries.values() if e.row is not None)

    # ---- persistence -----------------------------------------------------------
    def persist(self) -> None:
        """Write the catalog to its cache file now, if anything changed."""
        with self._lock:
            self._maybe_persist(force=True)

    def _maybe_persist(self, *, force: bool = False) -> None:
        if not self._dirty or self.cache_file is None:
            return
        now = time.monotonic()
        if (
            not force
            and self._persisted_at is not None
            and now - self._persisted_at < _PERSIST_EVERY_S
        ):
            return
        rows: list[list[Any]] = []
        bad: list[list[Any]] = []
        ordered = sorted(self._entries.items(), key=lambda kv: _order(kv[0]))
        for e in (entry for _k, entry in ordered):
            if e.row is not None:
                r = e.row
                rows.append([
                    r.rel, r.mtime_ns, r.size, e.seen_ns, r.id, r.type, r.title, r.slug,
                    r.source, r.url, r.program, r.severity, r.status, r.created,
                    list(r.tags), list(r.links),
                ])  # fmt: skip
            else:
                bad.append([self._rel(e.path), e.mtime_ns, e.size, e.seen_ns, e.error])
        payload = {
            "version": CATALOG_VERSION,
            "vault": str(self.vault),
            "rows": rows,
            "bad": bad,
        }
        try:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            write_bytes_atomic(self.cache_file, data.encode("utf-8", "replace"))
        except (OSError, ValueError, TypeError) as exc:
            log.debug("could not write the vault catalog %s: %s", self.cache_file, exc)
            return
        self._dirty = False
        self._persisted_at = now

    def _load(self) -> None:
        """Start from the cached file. Anything unexpected discards all of it."""
        if self.cache_file is None:
            return
        try:
            raw = self.cache_file.read_bytes()
        except OSError:
            return
        try:
            data = json.loads(raw)
            if data.get("version") != CATALOG_VERSION:
                return
            if _key(str(data.get("vault", ""))) != _key(self.vault):
                return
            for item in data["rows"]:
                (rel, mtime_ns, size, seen_ns, nid, ntype, title, slug, source, url, program,
                 severity, status, created, tags, links) = item  # fmt: skip
                path = self._cached_path(rel)
                row = CatalogRow(
                    path=path,
                    rel=str(rel),
                    mtime_ns=int(mtime_ns),
                    size=int(size),
                    id=_str(nid),
                    type=_str(ntype),
                    title=_str(title),
                    slug=_str(slug),
                    source=_opt(source),
                    url=_opt(url),
                    program=_opt(program),
                    severity=_opt(severity),
                    status=_opt(status),
                    created=_opt(created),
                    tags=tuple(_str(t) for t in tags),
                    links=tuple(_str(t) for t in links),
                )
                self._put(_key(path), _Entry(path, row.mtime_ns, row.size, int(seen_ns), row))
            for item in data.get("bad", []):
                rel, mtime_ns, size, seen_ns, error = item
                path = self._cached_path(rel)
                entry = _Entry(path, int(mtime_ns), int(size), int(seen_ns), None, _opt(error))
                self._put(_key(path), entry)
        except Exception as exc:  # noqa: BLE001 - advisory cache: rebuild from the vault
            log.debug("ignoring the vault catalog cache %s: %s", self.cache_file, exc)
            self._reset()

    def _rel(self, path: Path) -> str:
        return os.path.relpath(_key(path), _key(self.vault)).replace(os.sep, "/")

    def _cached_path(self, rel: object) -> Path:
        parts = str(rel).split("/")
        if not rel or any(p in ("", ".", "..") for p in parts) or ":" in str(rel):
            raise ValueError(f"bad cached path {rel!r}")
        return self.vault.joinpath(*parts)

    def _reset(self) -> None:
        self._entries.clear()
        for index in (self._by_id, self._by_slug, self._by_legacy, self._by_stem):
            index.clear()
        self._sorted = None


def _str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {type(value).__name__}")
    return value


def _opt(value: object) -> str | None:
    return None if value is None else _str(value)


# --------------------------------------------------------------------------------
# One catalog per vault per process
# --------------------------------------------------------------------------------

_CATALOGS: dict[str, VaultCatalog] = {}
_CATALOGS_LOCK = threading.Lock()
_atexit_registered = False


def _cache_file(vault: Path) -> Path | None:
    d = state_dir()
    return None if d is None else d / f"catalog-{vault_key(vault)}.json"


def get_catalog(vault: Path | str) -> VaultCatalog:
    """This process's catalog of `vault`, created on first use (from the cache file
    when there is one). Not refreshed here: call `ensure_fresh`."""
    global _atexit_registered
    vault = Path(vault)
    key = _key(vault)
    cache_file = _cache_file(vault)
    with _CATALOGS_LOCK:
        cat = _CATALOGS.get(key)
        if cat is None or cat.cache_file != cache_file:
            cat = VaultCatalog(vault, cache_file)
            _CATALOGS[key] = cat
            if not _atexit_registered:
                atexit.register(_persist_all)
                _atexit_registered = True
        return cat


def peek_catalog(vault: Path | str) -> VaultCatalog | None:
    """The catalog of `vault` if this process already has one; never builds one."""
    with _CATALOGS_LOCK:
        return _CATALOGS.get(_key(vault))


def fresh_catalog(vault: Path | str, *, max_age: float = 0.0) -> VaultCatalog:
    """`get_catalog` + `ensure_fresh(max_age)`: the usual way to read it."""
    cat = get_catalog(vault)
    cat.ensure_fresh(max_age=max_age)
    return cat


def clear_catalogs() -> None:
    """Drop every in-process catalog (tests). Cache files are left alone."""
    with _CATALOGS_LOCK:
        _CATALOGS.clear()


def _persist_all() -> None:
    with _CATALOGS_LOCK:
        cats = list(_CATALOGS.values())
    for cat in cats:
        with contextlib.suppress(Exception):  # interpreter shutdown: never raise
            cat.persist()


__all__ = [
    "CATALOG_VERSION",
    "CatalogRow",
    "VaultCatalog",
    "clear_catalogs",
    "fresh_catalog",
    "get_catalog",
    "peek_catalog",
]
