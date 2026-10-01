"""Shared plumbing for ingestion sources.

A source is anything that yields :class:`~sift.vault.notes.Note` objects.
`run_source` saves them to the vault, indexes them in batches, and records a
per-source summary in ``vault/_state.json`` (read by ``sift status`` and the MCP
``stats`` tool).

What `run_source` guarantees, because each of these once corrupted the corpus:

* **Identity before filename.** A note is upserted by id (`write_note`). Ids that
  are derived from the title alone (``research-``, ``writeup-``, ``top10-``) are
  resolved by URL first: two different articles that share a title share that id,
  and the second used to overwrite the first while the run reported success. The
  later article gets a stable URL-hashed id (`url_note_id`) instead, and the clash
  is counted in ``collisions``.
* **Re-ingesting costs nothing when nothing changed.** A note identical to the file
  that already holds it is neither rewritten nor re-embedded (``unchanged``).
* **Pruned stays pruned.** Ids and URLs in the tombstone ledger
  (`sift.tombstones`) are skipped, so a bulk re-ingest does not undo ``sift prune``.
* **An abort still indexes what was saved.** Notes already written are indexed and
  the run is recorded (``complete: false``) even when the source raises or the user
  presses Ctrl-C; the exception still propagates.
* **Nothing prints.** Diagnostics go through `logging` (stderr), never stdout.

Sources use `KnownNotes` (or `have_note`) to skip a fetch they have already paid
for, `url_note_id` to derive a collision-safe id themselves, and `safe_get` so one
malformed link cannot abort a run.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from slugify import slugify

from sift.config import get_settings
from sift.index.store import Store
from sift.pipeline import index_notes
from sift.vault.notes import (
    IdConflict,
    Note,
    canonical_url,
    describe_error,
    load_note,
    locate_note,
    same_document,
    state_dir,
    vault_key,
    write_lock,
    write_note,
    write_text_atomic,
)
from sift.vault.schema import Frontmatter

log = logging.getLogger(__name__)

STATE_FILE = "_state.json"
_CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)

# What sources use. `canonical_url` is re-exported from the vault layer so skip
# checks, id resolution and `same_document` share one URL identity rule.
__all__ = [
    "FETCH_ERRORS",
    "IngestResult",
    "KnownNotes",
    "canonical_url",
    "clean_text",
    "extract_cwes",
    "have_note",
    "id_family",
    "is_title_derived",
    "iter_limited",
    "keep_longer_body",
    "load_state",
    "maintain_index",
    "record_run",
    "run_source",
    "safe_get",
    "url_note_id",
]


def clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_cwes(*blobs: str | None) -> list[str]:
    out: list[str] = []
    for b in blobs:
        if not b:
            continue
        for m in _CWE_RE.findall(b):
            v = m.upper()
            if v not in out:
                out.append(v)
    return out


def _one_line(exc: BaseException) -> str:
    return describe_error(exc)


# --------------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------------

# What a per-item fetch can raise besides an HTTP status. `httpx.InvalidURL` is
# neither an HTTPError nor a ValueError, and an IDNA-invalid host (an emoji domain,
# a stray control character) surfaces as a ValueError via UnicodeError. Catching only
# httpx.HTTPError let one such link escape the source generator and end the run on
# every attempt, at the same entry.
FETCH_ERRORS: tuple[type[BaseException], ...] = (httpx.HTTPError, httpx.InvalidURL, ValueError)


def safe_get(
    client: httpx.Client,
    url: str,
    *,
    what: str = "",
    raise_for_status: bool = True,
    **kwargs: Any,
) -> httpx.Response | None:
    """`client.get(url)`, or None (logged on stderr) when it cannot be fetched.

    Use it for every per-item request in a source, so a 404, a timeout or a link
    httpx cannot even parse skips that one item instead of aborting the run.
    `raise_for_status=False` returns non-2xx responses too.
    """
    try:
        r = client.get(url, **kwargs)
        if raise_for_status:
            r.raise_for_status()
        return r
    except FETCH_ERRORS as exc:
        log.warning("%sfetch failed %r: %s", f"{what}: " if what else "", url, _one_line(exc))
        return None


# --------------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------------

# `base[:69] + "-" + 8 hex` is at most 78 characters, so the hash survives even the
# legacy 80-character slug cut that old index rows and links still carry.
_URL_ID_BASE_MAX = 69
_URL_ID_HASH_LEN = 8


def id_family(note_id: str) -> str:
    """The id's prefix before the first '-': 'research' for 'research-foo-1a2b3c4d'."""
    return str(note_id or "").partition("-")[0]


def is_title_derived(meta: Frontmatter) -> bool:
    """True when `meta.id` is ``<family>-<slugify(title, max_length=90)>``.

    Such an id has the title as its only identity, so two different articles with
    one title (two vendors' "Security Advisory", a blog reusing "Release notes") get
    the same id. `run_source` resolves these by URL before saving.
    """
    family, sep, rest = str(meta.id or "").partition("-")
    return bool(family and sep and rest) and rest == slugify(meta.title or "", max_length=90)


def url_note_id(base_id: str, url: str | None) -> str:
    """A collision-safe id for the article at `url`: ``<base_id[:69]>-<sha1[:8]>``.

    Deterministic in the canonical URL (`canonical_url`), so a re-run, or the same
    link seen with ``?utm_source=rss``, http instead of https or a trailing slash,
    maps back to the same id and the same file. Raises ValueError without a URL:
    there is nothing to disambiguate by.
    """
    cu = canonical_url(url)
    if not cu:
        raise ValueError("url_note_id needs a url")
    digest = hashlib.sha1(cu.encode("utf-8")).hexdigest()[:_URL_ID_HASH_LEN]
    base = str(base_id or "")[:_URL_ID_BASE_MAX].rstrip("-") or "note"
    return f"{base}-{digest}"


class KnownNotes:
    """What the vault already holds, for one ingest run's "do I already have this?"
    and "which id does this article get?" decisions.

    Built lazily, once, from the vault catalog (`sift.vault.catalog`: one stat walk,
    rows carry id and url), so a per-entry check is a dict lookup rather than a file
    probe. The old probes looked for ``<slug>.md`` after filenames had become titles,
    never matched, and made every run re-fetch and re-embed everything.

    URL keys are scoped to the id family (``research``, ``writeup``, ``top10``...):
    a Top-10 nomination of an article a research feed already carried is still
    recorded as its own note with its own tags. Cross-source dedup is a policy
    decision, not a skip.

    A source should call `add` for each note it yields, so a later entry of the same
    run (another feed carrying the same post) sees it. Not thread-safe for writers;
    one instance per run.
    """

    def __init__(self, vault: Path | None = None) -> None:
        self.vault = Path(vault) if vault is not None else get_settings().resolved_vault()
        self._lock = threading.RLock()
        self._cat: Any = None
        self._by_family: dict[tuple[str, str], list[str]] | None = None
        self._by_url: dict[str, list[str]] = {}
        self._run_urls: dict[str, set[str]] = {}  # id -> canonical urls added this run

    # ---- index -----------------------------------------------------------------
    def _ensure(self) -> dict[tuple[str, str], list[str]]:
        with self._lock:
            if self._by_family is None:
                from sift.vault.catalog import fresh_catalog

                cat = fresh_catalog(self.vault)
                by_family: dict[tuple[str, str], list[str]] = {}
                by_url: dict[str, list[str]] = {}
                for row in cat.rows():
                    cu = canonical_url(row.url)
                    if not cu:
                        continue
                    _append(by_family.setdefault((id_family(row.id), cu), []), row.id)
                    _append(by_url.setdefault(cu, []), row.id)
                self._cat = cat
                self._by_url = by_url
                self._by_family = by_family
            return self._by_family

    def add(self, meta: Frontmatter) -> None:
        """Record a note yielded or saved this run. A no-op until the index is built
        (it is built from the catalog, which already holds every note sift saved)."""
        cu = canonical_url(meta.url)
        if not cu or not meta.id:
            return
        with self._lock:
            if self._by_family is None:
                return
            _append(self._by_family.setdefault((id_family(meta.id), cu), []), meta.id)
            _append(self._by_url.setdefault(cu, []), meta.id)
            self._run_urls.setdefault(meta.id, set()).add(cu)

    # ---- lookups ---------------------------------------------------------------
    def ids_for_url(self, url: str | None, *, family: str | None = None) -> tuple[str, ...]:
        """Ids of notes holding this article (canonical URL), within `family` when given."""
        cu = canonical_url(url)
        if not cu:
            return ()
        by_family = self._ensure()
        with self._lock:
            if family is not None:
                return tuple(by_family.get((family, cu), ()))
            return tuple(self._by_url.get(cu, ()))

    def has_url(self, url: str | None, *, family: str | None = None) -> bool:
        """True when a note in `family` (any family when None) holds this article.

        The pre-fetch skip check for sources whose id is only known after the fetch
        (top10 takes its title from the page): ``known.has_url(link, family="top10")``.
        """
        return bool(self.ids_for_url(url, family=family))

    def has(self, meta: Frontmatter) -> bool:
        """True when the vault already holds the document `meta` describes.

        A title-derived id is judged by URL within its family: an id match alone may
        be a different article that happens to share the title, and treating that
        as "already have it" would skip the second article forever. Any other id is
        judged by `locate_note`: a verified file carrying the id that holds the same
        document (`same_document`), wherever the file is and whatever it is called.
        """
        if canonical_url(meta.url) and is_title_derived(meta):
            return self.has_url(meta.url, family=id_family(meta.id))
        return locate_note(self.vault, meta.id, meta=meta) is not None

    def resolve_id(self, meta: Frontmatter) -> str:
        """The id the article `meta` describes should be saved under.

        * Its URL is already on disk in the family: that note's id (a re-run, a
          tracking-parameter variant, or an upstream retitle all update the one note).
        * Its id is carried by a different article (another non-empty URL):
          `url_note_id(meta.id, meta.url)`, stable across runs.
        * Otherwise `meta.id`, unchanged. A note without a URL is never re-keyed.

        Pure lookup: call `add` once the note is saved or yielded.
        """
        return self._resolve(meta)[0]

    def _resolve(self, meta: Frontmatter) -> tuple[str, str]:
        """(id, how): how is "" (kept), "known" (the URL is on disk under that id) or
        "clash" (the id belongs to another article; a new URL-hashed id)."""
        cu = canonical_url(meta.url)
        if not cu or not meta.id:
            return meta.id, ""
        ids = self._ensure().get((id_family(meta.id), cu))
        if ids:
            return (meta.id, "") if meta.id in ids else (ids[0], "known")
        for other in self._owner_urls(meta.id):
            if other and other != cu:
                return url_note_id(meta.id, meta.url), "clash"
        return meta.id, ""

    def _owner_urls(self, note_id: str) -> set[str]:
        """Canonical URLs of every file carrying `note_id` ("" for a file without one)."""
        self._ensure()
        with self._lock:
            # The catalog follows sift's own writes; this also notices another
            # process's (at most a second stale, and the write path re-checks on disk).
            self._cat.ensure_fresh(max_age=1.0)
            urls = set(self._run_urls.get(note_id, ()))
            for row in self._cat.by_id(note_id):
                urls.add(canonical_url(row.url))
        return urls


def _append(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def have_note(vault: Path, meta: Frontmatter) -> bool:
    """True when the vault already holds the document `meta` describes (see
    `KnownNotes.has`), wherever its file is and whatever it is called.

    Lets a network source skip a fetch it has already paid for. A one-shot check:
    for title-derived ids it builds a URL index of the whole vault, so a source that
    checks every entry should keep one `KnownNotes` for the run instead.
    """
    return KnownNotes(vault).has(meta)


def keep_longer_body(existing: Note, incoming: Note) -> Note:
    """A `run_source` merge: take the incoming note, but never replace a body with a
    shorter one - a feed excerpt after a full-text fetch, or a body fetch that failed
    and fell back to the teaser. Frontmatter is still refreshed.

    Raises `IdConflict` when `existing` is a different document (see
    `same_document`): a merge must never graft one article onto another.
    """
    same, reason = same_document(existing.meta.to_yaml_dict(), incoming.meta)
    if not same:
        raise IdConflict(
            incoming.meta.id,
            existing.path or Path(existing.meta.id),
            reason,
            existing_source=existing.meta.source,
            existing_url=existing.meta.url,
        )
    if len(existing.body.strip()) > len(incoming.body.strip()):
        incoming.body = existing.body
    return incoming


def _renders_as(note: Note, rendered: str, ingested: datetime | None) -> bool:
    """Would writing `note` reproduce `rendered` (an existing note's render, taken
    before any merge touched it)? `ingested` is ignored: it says when sift wrote the
    file, not what the note says."""
    stamp = note.meta.ingested
    note.meta.ingested = ingested
    try:
        return note.render() == rendered
    finally:
        note.meta.ingested = stamp


# --------------------------------------------------------------------------------
# Results and the per-source state file
# --------------------------------------------------------------------------------


@dataclass
class IngestResult:
    source: str
    # New notes: files created this run. (`_state.json` keeps this key: `sift status`
    # reads it.)
    written: int = 0
    indexed_chunks: int = 0
    # Notes that failed to save, plus notes saved but whose index batch failed.
    errors: int = 0
    # Clashes that were disambiguated rather than silently overwritten: a new file
    # took `Title (n).md`, or an article got a URL-hashed id because a different
    # article holds its title-derived id. Counted once per note.
    collisions: int = 0
    # Existing notes rewritten because their content changed.
    updated: int = 0
    # Re-ingested notes identical to the file already on disk: not rewritten and,
    # when already indexed, not re-embedded.
    unchanged: int = 0
    # Skipped: pruned or forgotten (`sift.tombstones`).
    tombstoned: int = 0
    # Not saved: the id is held by a different document (`IdConflict`), e.g. a KEV
    # note for a CVE that NVD already wrote, with no merge for that source.
    id_conflicts: int = 0
    # Notes the source yielded.
    seen: int = 0
    # False when the source raised or the run was interrupted.
    complete: bool = True
    aborted: str | None = None

    @property
    def saved(self) -> int:
        """Notes written to disk this run (new + updated)."""
        return self.written + self.updated

    def summary(self) -> str:
        """One line for the CLI, e.g. '3 new, 1 updated, 40 unchanged, 12 chunks, 0 errors'."""
        parts = [f"{self.written} new", f"{self.updated} updated", f"{self.unchanged} unchanged"]
        for label, n in (
            ("tombstoned", self.tombstoned),
            ("collisions", self.collisions),
            ("id conflicts", self.id_conflicts),
        ):
            if n:
                parts.append(f"{n} {label}")
        parts += [f"{self.indexed_chunks} chunks", f"{self.errors} errors"]
        if not self.complete:
            parts.append(f"INCOMPLETE ({self.aborted or 'aborted'})")
        return ", ".join(parts)


def _state_path() -> Path:
    return get_settings().resolved_vault() / STATE_FILE


_STATE_WARNED: set[tuple[str, int, int]] = set()
_STATE_LOCK = threading.Lock()
_STATE_LOCK_TIMEOUT_S = 30.0


def _read_state(p: Path) -> tuple[dict, str]:
    """(state, status). status is "ok", "missing", "corrupt" (exists, but is not a
    JSON object) or "error" (could not be read right now, e.g. a Windows sharing
    violation - not evidence that the file is bad)."""
    try:
        text = p.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}, "missing"
    except UnicodeDecodeError:
        _warn_state(p, "not UTF-8 text")
        return {}, "corrupt"
    except OSError as exc:
        _warn_state(p, f"unreadable ({_one_line(exc)})")
        return {}, "error"
    try:
        data = json.loads(text)
    except ValueError as exc:
        _warn_state(p, f"not valid JSON ({type(exc).__name__})")
        return {}, "corrupt"
    if not isinstance(data, dict):
        _warn_state(p, "not a JSON object")
        return {}, "corrupt"
    return data, "ok"


def _warn_state(p: Path, why: str) -> None:
    try:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
    except OSError:
        key = (str(p), -1, -1)
    if key in _STATE_WARNED:
        return  # `stats` reads it on every MCP call
    _STATE_WARNED.add(key)
    log.warning("ingest state file %s is %s; ignoring it", p.name, why)


def load_state() -> dict:
    """Per-source summaries of the last ingest runs. Read-only (the MCP `stats` tool
    calls it): an unreadable file reads as {} and is never modified here."""
    return _read_state(_state_path())[0]


@contextlib.contextmanager
def _state_lock(vault: Path) -> Iterator[None]:
    """Serialise `record_run`'s read-modify-write across threads and processes, so
    two ingests finishing together cannot drop each other's entry. Degrades to the
    thread lock alone when no lock file can be taken: the file is display-only."""
    with _STATE_LOCK:
        lock = None
        d = state_dir()
        if d is not None:
            try:
                from filelock import FileLock

                lock = FileLock(
                    str(d / f"ingest-state-{vault_key(vault)}.lock"),
                    timeout=_STATE_LOCK_TIMEOUT_S,
                )
                lock.acquire()
            except Exception as exc:  # noqa: BLE001 - Timeout, read-only dir: carry on
                log.debug("ingest state lock unavailable: %s", exc)
                lock = None
        try:
            yield
        finally:
            if lock is not None:
                lock.release()


def record_run(
    source: str,
    result: IngestResult,
    *,
    complete: bool = True,
    aborted: str | None = None,
) -> None:
    """Record `result` as the latest run of `source` in ``_state.json``.

    Atomic (temp file + replace), so a crash or a concurrent reader never sees half
    a file, and locked, so two ingests cannot lose each other's entry. A corrupt file
    is moved aside to ``_state.json.corrupt-<stamp>`` rather than overwritten with a
    single entry. An incomplete run is recorded as such; ``last_complete`` keeps the
    time of the last run that finished.
    """
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with _state_lock(p.parent):
        for attempt in range(5):  # Windows: a reader mid-replace blocks us briefly
            state, status = _read_state(p)
            if status != "error":
                break
            time.sleep(0.05 * (attempt + 1))
        if status == "error":
            # Unreadable right now is not corrupt: writing would drop every other
            # source's entry, and moving it aside would be wrong.
            log.warning("not recording the %s run: %s could not be read", source, p.name)
            return
        if status == "corrupt":
            aside = p.with_name(f"{p.name}.corrupt-{datetime.now(UTC):%Y%m%d%H%M%S}")
            try:
                os.replace(p, aside)
            except OSError as exc:
                log.warning("not recording the %s run: %s is unreadable and could not be "
                            "moved aside: %s", source, p.name, _one_line(exc))  # fmt: skip
                return
            log.warning("moved the unreadable %s aside to %s", p.name, aside.name)
        prev = state.get(source)
        prev = prev if isinstance(prev, dict) else {}
        if complete:
            last_complete = now
        elif prev.get("complete", True) is not False and prev.get("last_run"):
            last_complete = prev.get("last_run")
        else:
            last_complete = prev.get("last_complete")
        entry: dict[str, Any] = {
            "last_run": now,
            "written": result.written,
            "updated": result.updated,
            "unchanged": result.unchanged,
            "tombstoned": result.tombstoned,
            "collisions": result.collisions,
            "id_conflicts": result.id_conflicts,
            "errors": result.errors,
            "complete": complete,
        }
        if last_complete:
            entry["last_complete"] = last_complete
        if aborted:
            entry["aborted"] = aborted
        state[source] = entry
        write_text_atomic(p, json.dumps(state, indent=2))


# --------------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------------

Merge = Callable[[Note, Note], Note | None]


def _label(note: object) -> str:
    meta = getattr(note, "meta", None)
    return str(getattr(meta, "id", None) or "<not a note>")


def maintain_index(store: Any, source_name: str) -> None:
    """Routine index maintenance (`Store.optimize`) after a run that wrote rows, for
    every ingest path. Never raises: the notes are already saved and indexed, and a
    failed compaction must not hide that; a failure is logged on stderr."""
    optimize = getattr(store, "optimize", None)  # a stubbed store may have none
    if not callable(optimize):
        return
    try:
        report = optimize()
        if isinstance(report, dict) and report.get("error"):
            log.warning("%s: index maintenance failed: %s", source_name, report["error"])
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "%s: index maintenance failed (notes are saved): %s", source_name, _one_line(exc)
        )


def run_source(
    source_name: str,
    notes: Iterable[Note],
    *,
    reindex_fts: bool = True,
    flush_every: int = 200,
    on_progress: Callable[[int, str], None] | None = None,
    id_by_url: bool | None = None,
    merge: Merge | None = None,
    skip_tombstoned: bool = True,
) -> IngestResult:
    """Save each note to the vault, then index in batches of ``flush_every``.

    Per note, in order:

    1. **Tombstones.** A note whose id or URL is in the ledger for its source is
       skipped (``tombstoned``) unless ``skip_tombstoned=False``.
    2. **Identity.** With ``id_by_url`` None (default) a title-derived id
       (`is_title_derived`) is resolved by URL (`KnownNotes.resolve_id`); True
       resolves every note that has a URL, False none. Never use True for ids that
       must stay fixed (a CVE number).
    3. **Merge.** With ``merge``, a file already carrying the id is loaded and
       ``merge(existing, incoming)`` decides what is written there: a Note (saved to
       that file, identity check off: the merge has decided it is the same
       document), None (leave it as it is), or raise `IdConflict`.
    4. **Unchanged.** A note identical to the file holding it is not rewritten
       (``unchanged``); it is only queued for indexing when its id is missing from
       the index.
    5. **Write** (`write_note`, upsert by id). ``written`` counts new files,
       ``updated`` rewrites. A different document under the same id is refused and
       counted in ``id_conflicts`` - except that a title-derived id is then retried
       once under its URL-hashed id.

    Which file holds a note that is not resolved by URL (steps 3-5): its own ``path``
    when that file carries its id - the source chose it, e.g. a source-side merge into
    one of two twins - else the file `locate_note` finds.

    The whole step runs under the vault write lock. `on_progress` receives (notes
    the source yielded so far, latest title): sources are generators over the
    network, so the total is unknowable up front.

    If the source raises or the run is interrupted, the notes already saved are
    still indexed and the run is recorded with ``complete: false``; the exception
    then propagates. Index maintenance (`Store.optimize`) runs once at the end when
    ``reindex_fts`` is set and anything was indexed.
    """
    s = get_settings()
    vault = s.resolved_vault()
    store = Store()
    res = IngestResult(source=source_name)
    pending: list[Note] = []
    queued = 0
    known = KnownNotes(vault)
    indexed: set[str] | None = None
    indexed_loaded = False

    tomb = None
    if skip_tombstoned:
        from sift.tombstones import load_tombstones

        tomb = load_tombstones()  # an unreadable ledger raises: never ingest past it

    def flush() -> None:
        if not pending:
            return
        batch = list(pending)
        pending.clear()  # before indexing: a Ctrl-C mid-batch must not re-embed it in `finally`
        try:
            res.indexed_chunks += index_notes(batch, store)
        except Exception as exc:  # noqa: BLE001 - the notes are on disk; `sift reindex` picks them up
            res.errors += len(batch)
            log.warning(
                "%s: index batch of %d notes failed (they are saved; run `sift reindex`): %s",
                source_name,
                len(batch),
                _one_line(exc),
            )
        if not on_progress:  # the caller is rendering a bar; this would tear it
            log.info("  .. %s: %d notes saved", source_name, res.saved)

    def queue(note: Note) -> None:
        nonlocal queued
        pending.append(note)
        queued += 1

    def is_indexed(note_id: str) -> bool:
        """Whether the index has rows for `note_id`; True when that cannot be told
        (a stubbed or broken store), so an unknown never re-embeds the vault."""
        nonlocal indexed, indexed_loaded
        if not indexed_loaded:
            indexed_loaded = True
            try:
                indexed = set(store.indexed_ids())
            except Exception as exc:  # noqa: BLE001
                log.debug("%s: indexed ids unavailable: %s", source_name, exc)
                indexed = None
        return indexed is None or note_id in indexed

    def tombstoned(meta: Frontmatter) -> bool:
        return tomb is not None and (
            tomb.has_id(meta.id, source=meta.source) or tomb.has_url(meta.url, source=meta.source)
        )

    def resolves(meta: Frontmatter) -> bool:
        if id_by_url is None:
            return is_title_derived(meta)
        return id_by_url

    def load_or_none(path: Path | None) -> Note | None:
        if path is None:
            return None
        try:
            return load_note(path)
        except Exception as exc:  # noqa: BLE001 - unreadable right now: write_note decides
            log.debug("%s: could not load %s: %s", source_name, path, exc)
            return None

    def url_carrier(note_id: str, cu: str) -> tuple[Path | None, Note | None]:
        """The file carrying `note_id`, preferring the one that holds the URL `cu`.
        An id can already be spread over several files (duplicates written before
        upsert-by-id), and this article's own copy is the one to update."""
        first = locate_note(vault, note_id)  # verified on disk; refreshes the catalog
        if first is None:
            return None, None
        from sift.vault.catalog import get_catalog

        for row in get_catalog(vault).by_id(note_id):
            if canonical_url(row.url) == cu:
                held = load_or_none(row.path)
                if (
                    held is not None
                    and held.meta.id == note_id
                    and canonical_url(held.meta.url) == cu
                ):
                    return row.path, held
        return first, load_or_none(first)

    def chosen_carrier(note: Note) -> tuple[Path | None, Note | None]:
        """``note.path`` when that file carries the note's id: the source picked the
        file (a source-side merge into one of several twins), so it is the one to
        update - not whichever twin `locate_note` ranks first, which would get the
        other twin's merged content written over its own."""
        if note.path is None:
            return None, None
        held = load_or_none(Path(note.path))
        if held is None or held.meta.id != note.meta.id:
            return None, None
        return Path(note.path), held

    def ingest_one(note: Note) -> None:
        meta = note.meta
        if tombstoned(meta):
            res.tombstoned += 1
            return
        base_id = meta.id
        cu = canonical_url(meta.url)
        by_url = bool(cu) and resolves(meta)
        clash = False

        def rekey(why: str) -> bool:
            """Give this article its URL-hashed id; False if it already has it."""
            nonlocal clash
            new_id = url_note_id(base_id, meta.url)
            if new_id == meta.id:
                return False
            log.warning("%s: %s; saving %s as %s", source_name, why, meta.url, new_id)
            meta.id = new_id
            clash = True
            return True

        if by_url:
            new_id, how = known._resolve(meta)
            if how == "clash":
                rekey(f"id {base_id} belongs to another article")
            elif new_id != meta.id:
                meta.id = new_id  # this article is already on disk under that id
            if meta.id != base_id and tombstoned(meta):
                res.tombstoned += 1
                return

        with write_lock(vault):
            verify = True
            existing_path: Path | None = None
            existing: Note | None = None
            for _attempt in range(2):
                if by_url:
                    existing_path, existing = url_carrier(meta.id, cu)
                else:
                    existing_path, existing = chosen_carrier(note)
                    if existing_path is None:
                        # With a merge any carrier counts (the merge judges identity);
                        # without one, only a file holding this same document.
                        existing_path = locate_note(
                            vault, meta.id, meta=None if merge is not None else meta
                        )
                        existing = load_or_none(existing_path)
                if by_url and existing is not None:
                    # Checked on disk, whatever the resolver's index said: a file that
                    # carries this id but holds another URL is another article, even
                    # when its title and host match (a blog reusing "Release notes").
                    held = canonical_url(existing.meta.url)
                    if held and held != cu:
                        if not rekey(f"id {meta.id} is held by another article"):
                            raise IdConflict(
                                meta.id,
                                existing_path or Path(meta.id),
                                "url",
                                existing_source=existing.meta.source,
                                existing_url=existing.meta.url,
                            )
                        existing_path = existing = None
                        continue
                    if merge is None and not same_document(existing.meta.to_yaml_dict(), meta)[0]:
                        existing_path = existing = None  # not this document: write_note decides
                break

            def unchanged() -> None:
                res.unchanged += 1
                if not is_indexed(meta.id):
                    # Index the file as it is on disk (a merge may have touched the
                    # loaded copy before declining).
                    fresh = load_or_none(existing_path)
                    if fresh is not None:
                        queue(fresh)

            if existing is not None:
                # Taken before a merge can mutate the loaded note.
                baseline = (existing.render(), existing.meta.ingested)
                if merge is not None:
                    merged = merge(existing, note)
                    if merged is None:
                        unchanged()
                        return
                    note, meta, verify = merged, merged.meta, False
                    meta.ingested = None  # a changed merge is a new write: stamp it now
                if _renders_as(note, *baseline):
                    unchanged()
                    return

            try:
                result = write_note(vault, note, existing=existing_path, verify_identity=verify)
            except IdConflict:
                # A title-derived id held by a different document (the holder has no
                # URL to compare, or comes from another source): this article takes
                # its own URL-hashed id. Anything else is the caller's conflict.
                if not by_url or not rekey(f"id {meta.id} is held by another document"):
                    raise
                result = write_note(vault, note, verify_identity=verify)

        if result.created:
            res.written += 1
        else:
            res.updated += 1
        if clash or result.title_clash:
            res.collisions += 1
            if result.title_clash:
                log.warning("%s: title clash, saved as %s", source_name, result.path.name)
        known.add(meta)
        queue(note)

    completed = False
    interrupted = False
    aborted: str | None = None
    try:
        for note in notes:
            res.seen += 1
            try:
                ingest_one(note)
            except IdConflict as exc:
                res.id_conflicts += 1
                log.warning("%s: %s", source_name, exc)
            except Exception as exc:  # noqa: BLE001 - one bad note never stops a run
                res.errors += 1
                log.warning("%s: failed on %s: %s", source_name, _label(note), _one_line(exc))
            if on_progress:
                on_progress(res.seen, str(getattr(getattr(note, "meta", None), "title", "")))
            if len(pending) >= flush_every:
                flush()
        completed = True
    except BaseException as exc:
        aborted = _one_line(exc)
        interrupted = isinstance(exc, KeyboardInterrupt)
        log.warning("%s: run aborted after %d notes: %s", source_name, res.seen, aborted)
        raise
    finally:
        # Index whatever was saved, including on Ctrl-C or a source error: up to
        # flush_every-1 notes used to sit on disk, invisible to search.
        flush()
        res.complete = completed
        res.aborted = aborted
        # Maintenance can wait for the next run when the user asked to stop.
        if reindex_fts and queued and not interrupted:
            maintain_index(store, source_name)
        try:
            record_run(source_name, res, complete=completed, aborted=aborted)
        except Exception as exc:  # noqa: BLE001 - never mask the run's own outcome
            log.warning("%s: could not record the run: %s", source_name, _one_line(exc))
    return res


def iter_limited(it: Iterator, limit: int | None) -> Iterator:
    if limit is None:
        yield from it
        return
    for i, x in enumerate(it):
        if i >= limit:
            return
        yield x
