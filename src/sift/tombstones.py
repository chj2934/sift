"""Ledger of pruned and forgotten notes, so bulk re-ingest can't bring them back.

Without it, `sift prune --yes` was undone by the next bulk ingest: `sift ingest nvd`
or `h1-public` re-yields every record, writing back and re-embedding exactly the
notes prune had just dropped. Prune (`prune.apply_prune`) and any "forget" action
record each removed note's id and url here. `ingest.base.run_source` checks the
ledger before saving a note.

The ledger is JSON lines in the index directory (`<SIFT_DB_PATH>/tombstones.jsonl`),
never inside the vault's content dirs, and survives `reindex --force` (which drops
the table, not the directory). One line per tombstone:

    {"id": "CVE-2019-1234", "url": "https://nvd.nist.gov/...", "source": "nvd",
     "reason": "sift prune", "ts": "2026-10-01T12:00:00+00:00"}

**Source scoping.** A tombstone recorded with a `source` blocks only that source.
Prune's verdict on an NVD CVE says nothing about the same CVE arriving later from
CISA KEV, and becoming known-exploited is exactly the signal that should bring it
back. KEV and NVD share the CVE id and the NVD url, so an unscoped check would bury
it forever. A tombstone recorded without a source (a user "forget") blocks every
source. Callers should pass the incoming note's source:
``tomb.has_id(note.meta.id, source=note.meta.source)``.

Writes are atomic (temp file in the same dir, then `os.replace`), so the ledger is
never left half-written. They are serialised by a process-wide lock plus a `filelock`
for other processes, and they never drop an existing line, including one that no
longer parses (it is skipped on load with a warning, not rewritten away).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from sift.config import get_settings
from sift.vault.notes import canonical_url

log = logging.getLogger(__name__)

LEDGER_NAME = "tombstones.jsonl"
_LOCK = threading.Lock()
_IO_ATTEMPTS = 6  # Windows: a reader holding the file blocks os.replace for a moment


def tombstones_path() -> Path:
    """The ledger file. It lives under the index dir, not the vault."""
    return get_settings().resolved_db() / LEDGER_NAME


def normalize_url(url: str | None) -> str:
    """Canonical form for matching: one resource, one key.

    The vault's own article identity rule (`sift.vault.notes.canonical_url`), the one
    ingest uses to tell whether it already holds an article: scheme, a leading
    ``www.``, the fragment, tracking parameters (``utm_*`` and the like) and a
    trailing slash are ignored, the host is lower-cased, and every other query
    parameter is kept, because `?id=1` and `?id=2` are different pages. A rule of its
    own here let a forgotten article come back through a feed that adds
    ``?utm_source=rss``. The ledger stores urls as given and normalises on load, so a
    rule change applies to existing lines too.
    """
    return canonical_url(url)


def _norm_source(source: object) -> str | None:
    s = str(source).strip().lower() if source is not None else ""
    return s or None


def _norm_id(note_id: object) -> str:
    return str(note_id).strip() if note_id is not None else ""


class Tombstones:
    """Read-only snapshot of the ledger. Ids match exactly; urls match normalised."""

    __slots__ = ("_ids", "_urls", "_count")

    def __init__(self, entries: Iterable[dict] = ()) -> None:
        ids: dict[str, set[str | None]] = {}
        urls: dict[str, set[str | None]] = {}
        count = 0
        for e in entries:
            src = _norm_source(e.get("source"))
            nid = _norm_id(e.get("id"))
            url = normalize_url(e.get("url"))
            if nid:
                ids.setdefault(nid, set()).add(src)
            if url:
                urls.setdefault(url, set()).add(src)
            count += bool(nid or url)
        self._ids = ids
        self._urls = urls
        self._count = count

    @staticmethod
    def _blocks(sources: set[str | None] | None, source: str | None) -> bool:
        if not sources:
            return False
        want = _norm_source(source)
        # No source asked for: any tombstone counts. A tombstone without a source
        # (a user "forget") blocks every source.
        return want is None or None in sources or want in sources

    def has_id(self, note_id: str | None, source: str | None = None) -> bool:
        """True if `note_id` is tombstoned for `source` (or for any source, if None)."""
        return self._blocks(self._ids.get(_norm_id(note_id)), source)

    def has_url(self, url: str | None, source: str | None = None) -> bool:
        """True if `url` is tombstoned for `source` (or for any source, if None)."""
        key = normalize_url(url)
        return bool(key) and self._blocks(self._urls.get(key), source)

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return f"Tombstones({self._count} entries)"


# --------------------------------------------------------------------------- #
# file I/O
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    with _LOCK:
        try:
            from filelock import FileLock
        except ImportError:  # pragma: no cover - shipped with huggingface_hub
            yield
            return
        with FileLock(str(path) + ".lock", timeout=60):
            yield


def _retry(fn):
    for attempt in range(_IO_ATTEMPTS):
        try:
            return fn()
        except PermissionError:
            if attempt == _IO_ATTEMPTS - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
    return None  # unreachable


def _read_text(path: Path) -> str:
    try:
        return _retry(lambda: path.read_text(encoding="utf-8", errors="replace"))
    except FileNotFoundError:
        return ""


def _parse(text: str) -> list[tuple[str, dict | None]]:
    """(raw line, entry) pairs; entry is None for a line that doesn't parse."""
    out: list[tuple[str, dict | None]] = []
    bad = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            obj = None
        if not isinstance(obj, dict):
            obj = None
            bad += 1
        out.append((line, obj))
    if bad:
        log.warning("tombstones: %d unreadable ledger line(s) ignored (kept on disk)", bad)
    return out


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _retry(lambda: os.replace(tmp, path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _key(e: dict) -> tuple[str, str, str | None]:
    return (_norm_id(e.get("id")), normalize_url(e.get("url")), _norm_source(e.get("source")))


def _entry(
    note_id: object = None,
    url: object = None,
    source: object = None,
    reason: str = "",
    ts: str = "",
) -> dict | None:
    nid = _norm_id(note_id)
    u = str(url).strip() if url else ""
    if not nid and not u:
        return None
    e: dict = {}
    if nid:
        e["id"] = nid
    if u:
        e["url"] = u
    src = _norm_source(source)
    if src:
        e["source"] = src
    if reason:
        e["reason"] = reason
    e["ts"] = ts
    return e


def _add(entries: list[dict]) -> int:
    if not entries:
        return 0
    path = tombstones_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        old = _read_text(path)
        seen = {_key(e) for _, e in _parse(old) if e is not None}
        added: list[dict] = []
        for e in entries:
            k = _key(e)
            if k in seen:
                continue
            seen.add(k)
            added.append(e)
        if added:
            text = old if not old or old.endswith("\n") else old + "\n"
            text += "".join(json.dumps(e, ensure_ascii=True) + "\n" for e in added)
            _atomic_write(path, text)
    return len(added)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def load_tombstones() -> Tombstones:
    """Snapshot of the ledger. Empty when there is none yet.

    Unreadable lines are skipped with a warning. An unreadable *file* raises: an
    ingest that silently ignored the ledger would bring back every pruned note.
    """
    return Tombstones(e for _, e in _parse(_read_text(tombstones_path())) if e is not None)


def record_tombstones(
    ids: Iterable[str] = (),
    urls: Iterable[str] = (),
    *,
    source: str | None = None,
    reason: str = "",
) -> int:
    """Tombstone each id and each url (one ledger line apiece); return how many were new.

    `source=None` (the default) blocks them for every source, which is right for a
    user "forget". Prune records per-source tombstones through
    :func:`record_note_tombstones` instead.
    """
    ts = _now()
    new = [_entry(note_id=i, source=source, reason=reason, ts=ts) for i in ids]
    new += [_entry(url=u, source=source, reason=reason, ts=ts) for u in urls]
    return _add([e for e in new if e is not None])


def record_note_tombstones(notes: Iterable, *, reason: str = "") -> int:
    """One ledger line per note, carrying its id, url and source together.

    Scoped to each note's own `source` (see the module docstring), so an NVD CVE
    pruned today still comes back if CISA KEV catalogues it.
    """
    ts = _now()
    new = [
        _entry(note_id=n.meta.id, url=n.meta.url, source=n.meta.source, reason=reason, ts=ts)
        for n in notes
    ]
    return _add([e for e in new if e is not None])


def remove_tombstones(ids: Iterable[str] = (), urls: Iterable[str] = ()) -> int:
    """Clear every tombstone (any source) on these ids or urls; return how many went.

    The undo for a prune (`prune.restore_quarantine` calls it) or a mistaken forget.
    Lines that don't parse are kept.
    """
    drop_ids = {_norm_id(i) for i in ids} - {""}
    drop_urls = {normalize_url(u) for u in urls} - {""}
    if not drop_ids and not drop_urls:
        return 0
    path = tombstones_path()
    if not path.exists():
        return 0
    with _locked(path):
        kept: list[str] = []
        removed = 0
        for line, e in _parse(_read_text(path)):
            if e is not None and (
                _norm_id(e.get("id")) in drop_ids or normalize_url(e.get("url")) in drop_urls
            ):
                removed += 1
                continue
            kept.append(line)
        if removed:
            _atomic_write(path, "".join(f"{line}\n" for line in kept))
    return removed
