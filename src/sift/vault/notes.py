"""Read / write / enumerate vault notes.

Every vault write goes through `write_note` (or `save_note`, its Path-returning
wrapper), which keeps four promises:

* **Upsert by id.** `meta.id` is a note's identity, never its filename. When a file
  already carries the id it is rewritten in place, in whatever folder it sits. A
  title change renames it, within that same folder, only when sift itself named
  the file after the old title; a filename the user chose in Obsidian is kept.
  Before this, a retitled, moved or re-ingested note got a *second* file: 76 ids
  were spread over 152 files in the real vault, and the index flipped between the
  copies on every reindex.
* **Same id is not proof of same document.** Title-derived ids collide once they are
  truncated, and KEV and NVD both key a note on the bare CVE number. When the file
  found by id was written by a different source, or holds a different URL under a
  different title, the write is refused with `IdConflict` instead of silently
  replacing another document (CLAUDE.md, "Slug collisions").
* **Atomic.** Content is written to a dot-prefixed temp file in the target folder
  and published with `os.replace` (rewrite) or a no-clobber rename (new file). A
  crash, a full disk or one bad character can no longer leave a 0-byte or half
  written note, and the previous version survives until the new one is complete.
* **Serialised.** A process-wide lock plus a cross-process file lock under the DB
  dir make "is this name free?" and "write it" one step, for the CLI and the MCP
  server alike.

Nothing in this module prints: the MCP server's stdout is its JSON-RPC channel.
Diagnostics go through `logging`, which reaches stderr even unconfigured.
"""

from __future__ import annotations

import contextlib
import errno
import glob
import hashlib
import json
import logging
import os
import re
import stat as _stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import frontmatter
import yaml
from slugify import slugify

from sift.vault.schema import REQUIRED_KEYS, Frontmatter

log = logging.getLogger(__name__)

# `[[Target]]`, `[[Target|alias]]`, `[[Target#Heading]]`, `[[Target#^block|alias]]`,
# `[[folder/Target]]`, `[[Target.md]]`. A heading or block part used to make the
# whole link fail to match, so every section-level citation (the normal Obsidian way
# to cite a specific part of a note) vanished from the link graph. `[[#Heading]]`
# points into the same note and is deliberately not matched.
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")

# Index rows and links written before slugs became uncapped carry this cut.
LEGACY_SLUG_MAX = 80

# Soft-deleted notes go here. Obsidian uses the same folder for its own trash, and
# every vault walker skips dot-directories.
TRASH_DIR = ".trash"


def link_target_slug(raw: str) -> str:
    """Slug of a wikilink target, with a folder prefix and a `.md` suffix dropped."""
    target = raw.strip().rsplit("/", 1)[-1].strip()
    if target.lower().endswith(".md"):
        target = target[:-3]
    return slugify(target)


def note_slug(note_id: str, title: str = "") -> str:
    """A note's slug: its whole id, slugified.

    It used to be cut at 80 characters, so ids sharing an 80-character prefix (long
    writeup titles, repeated `remember` titles) shared a slug, and `get_note` could
    only ever return one of each pair. Filenames are titles now, so nothing needs the
    cap. Index rows keep the old form until the next `reindex --force`; resolve those
    through `legacy_slug`.
    """
    return slugify(note_id or "") or slugify(title or "")


def legacy_slug(note_id: str, title: str = "") -> str:
    """The pre-uncapped slug (cut at 80 characters).

    Index rows written before the next `reindex --force`, and `[[links]]` or
    frontmatter `links:` copied from older tool results, still carry it.
    """
    return slugify(note_id or "", max_length=LEGACY_SLUG_MAX) or slugify(
        title or "", max_length=LEGACY_SLUG_MAX
    )


@dataclass
class Note:
    meta: Frontmatter
    body: str
    path: Path | None = None  # set once written / loaded
    # st_mtime of `path`, taken *before* the file was read (`load_note`) or right
    # after it was written (`write_note`). Indexing should store this rather than
    # stat again later: an edit that lands while the note is being embedded then
    # leaves the index older than the file, so the next incremental reindex picks
    # it up instead of skipping it forever.
    mtime: float | None = None

    # ---- derived ----
    @property
    def slug(self) -> str:
        return note_slug(self.meta.id, self.meta.title)

    @property
    def legacy_slug(self) -> str:
        return legacy_slug(self.meta.id, self.meta.title)

    @property
    def wikilinks(self) -> list[str]:
        """[[links]] found in the body, as slugs."""
        return [s for m in WIKILINK_RE.finditer(self.body) if (s := link_target_slug(m.group(1)))]

    def all_links(self) -> list[str]:
        seen: dict[str, None] = {}
        for raw in [*self.meta.links, *self.wikilinks]:
            s = link_target_slug(raw)
            if s:
                seen.setdefault(s, None)
        return list(seen)

    def render(self) -> str:
        fm = yaml.safe_dump(
            self.meta.to_yaml_dict(),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ).strip()
        body = self.body.strip()
        return f"---\n{fm}\n---\n\n{body}\n"


# Characters Windows forbids in a filename, plus control chars. Everything else -
# spaces, case, accents, CJK - is kept, because the filename IS the note's name in
# Obsidian and `[[Cookie sandwich - reading HttpOnly cookies]]` should resolve there
# exactly as it reads.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WS = re.compile(r"\s+")
# Windows refuses these as filenames whatever the extension.
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
# Vault path + type dir eat ~60 chars of the 260-char Windows limit; leave room.
FILENAME_MAX = 150


def title_to_filename(title: str, fallback: str = "note") -> str:
    """A human-readable, filesystem-safe filename stem for a note title."""
    stem = _ILLEGAL.sub("-", title or "")
    stem = _WS.sub(" ", stem).strip().strip(".")
    if len(stem) > FILENAME_MAX:
        stem = stem[:FILENAME_MAX].rsplit(" ", 1)[0].strip() or stem[:FILENAME_MAX]
    if stem.upper() in _RESERVED:
        stem = f"{stem}-note"
    return stem or slugify(fallback, max_length=80) or "note"


def note_path(vault: Path, meta: Frontmatter) -> Path:
    """Where a *new* note goes. The filename is its title, so the vault opens cleanly
    in Obsidian; `meta.id` remains the stable identity used by the index. An existing
    note stays wherever its file already is (see `write_note`)."""
    return Path(vault) / meta.type / f"{title_to_filename(meta.title, meta.id)}.md"


# --------------------------------------------------------------------------------
# Document identity
# --------------------------------------------------------------------------------

# Query parameters that only say where a reader came from.
_TRACKING_PARAMS = frozenset({"ref", "source", "fbclid", "gclid", "mc_cid", "mc_eid", "igshid"})


def canonical_url(url: object) -> str:
    """A comparison key for "is this the same article?".

    Scheme, a leading ``www.``, the fragment, ``utm_*`` and other tracking parameters
    and a trailing slash are dropped and the host is lowercased, so one article seen
    over http and https, or through a feed that adds ``?utm_source=rss``, compares
    equal. Every other query parameter is kept (sorted): some sites identify the
    article by one, e.g. WordPress ``?p=123``. Returns "" for no URL.
    """
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw if "://" in raw or raw.startswith("//") else f"//{raw}")
        host = (parts.hostname or "").lower().removeprefix("www.")
        port = parts.port
    except ValueError:
        return raw
    if not host:
        return raw.rstrip("/")
    netloc = f"{host}:{port}" if port and port not in (80, 443) else host
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    )
    out = netloc + parts.path.rstrip("/")
    return f"{out}?{urlencode(query)}" if query else out


def _norm_source(value: object) -> str:
    return str(value or "").strip().casefold()


def _norm_title(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _id_of(metadata: Mapping[str, Any] | None) -> str | None:
    if not metadata:
        return None
    value = metadata.get("id")
    return None if value is None else str(value)


def same_document(on_disk: Mapping[str, Any], meta: Frontmatter) -> tuple[bool, str]:
    """Whether a file whose frontmatter is `on_disk` holds the document `meta` describes.

    Returns ``(same, reason)``; `reason` names what differs: "id", "source" or "url".
    The rule:

    * the id and the source must match - KEV and NVD both key on the CVE number, and
      a hand-written note never belongs to an ingest source;
    * when both sides carry a URL, the canonical URLs must match *or* the titles
      must. A changed URL alone is not a different document (Chromium doc URLs embed
      the revision, so they change on every sync), and a retitle under the same URL
      is the same document (NVD edits its descriptions). Different URL and different
      title - two long article titles that truncate to one id - is a different
      document;
    * when either side has no URL (`remember`, hand-written and tool notes), id and
      source decide, so a retitle is an update.

    Today's save already overwrote whenever the title path held the same id, so this
    only ever refuses writes it used to allow, plus the retitles it now upserts on
    purpose instead of forking a second file.
    """
    if _id_of(on_disk) != meta.id:
        return False, "id"
    if _norm_source(on_disk.get("source")) != _norm_source(meta.source):
        return False, "source"
    a, b = canonical_url(on_disk.get("url")), canonical_url(meta.url)
    if not a or not b or a == b:
        return True, ""
    if _norm_title(on_disk.get("title")) == _norm_title(meta.title):
        return True, ""
    return False, "url"


class IdConflict(ValueError):
    """`meta.id` is already carried by a file holding a different document.

    Nothing was written. The caller decides: give the incoming note a distinct id
    (e.g. a URL-hash suffix), merge it into the existing note (load `path`, merge,
    save with ``existing=path``), or count it as an error.
    """

    def __init__(
        self,
        note_id: str,
        path: Path,
        reason: str,
        *,
        existing_source: str | None = None,
        existing_url: str | None = None,
    ) -> None:
        self.note_id = note_id
        self.path = path
        self.reason = reason
        self.existing_source = existing_source
        self.existing_url = existing_url
        super().__init__(
            f"id {note_id!r} is already on disk at {path.name} "
            f"(source {existing_source or '-'}) as a different document "
            f"({reason} differs); not overwritten"
        )


class NotANote(ValueError):
    """The file has no frontmatter: empty (what a crashed write used to leave), a
    landing page, or a hand-written note waiting for `sift ingest local`."""


@dataclass(frozen=True)
class SaveResult:
    """What `write_note` did."""

    path: Path  # where the note now lives
    created: bool  # no file carried the id: a new file was written
    previous_path: Path | None  # the file that carried the id (None when created)
    # A new file took `Title (n).md` because a different note holds `Title.md`. This,
    # not "path != note_path()", is a title clash: an existing note found by id may
    # legitimately live under another name.
    title_clash: bool
    # Other files that also carry this id (pre-existing duplicates); left untouched.
    duplicates: tuple[Path, ...] = ()

    @property
    def renamed(self) -> bool:
        return self.previous_path is not None and self.previous_path != self.path


# --------------------------------------------------------------------------------
# Low-level atomic file operations
# --------------------------------------------------------------------------------

_RETRIES = 6  # ~1.5 s in total
_CLAIM_ATTEMPTS = 64


def _retrying(fn: Any, *args: Any) -> Any:
    """Run `fn`, retrying a Windows sharing violation briefly.

    Obsidian, an AV scanner or another sift process reading the file holds a handle
    without FILE_SHARE_DELETE for a few milliseconds; os.replace / os.rename onto or
    from it fail with PermissionError until it is closed.
    """
    for attempt in range(_RETRIES):
        try:
            return fn(*args)
        except PermissionError:
            if os.name != "nt" or attempt == _RETRIES - 1:
                raise
            time.sleep(0.05 * 2**attempt)
    return None  # pragma: no cover - the loop always returns or raises


def _write_temp(directory: Path, stem: str, data: bytes, *, durable: bool = False) -> Path:
    """Write `data` to a fresh, uniquely named temp file in `directory`.

    The name starts with a dot and does not end in `.md`, so neither a vault walker
    nor Obsidian ever sees it, and it is unique per call - two threads saving the same
    note can never publish each other's half-written temp file.
    """
    fd, name = tempfile.mkstemp(dir=directory, prefix=f".{stem[:40]}.", suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            if durable:  # power-loss safety; a process kill is already covered
                fh.flush()
                os.fsync(fh.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return tmp


def _replace_into(tmp: Path, dest: Path) -> None:
    """Atomically publish `tmp` over `dest` (which may or may not exist)."""
    _retrying(os.replace, tmp, dest)


def _move_no_clobber(src: Path, dest: Path) -> None:
    """Move `src` to `dest`, raising FileExistsError if `dest` exists.

    Atomic in both senses that matter: `dest` never exists half-written, and a file
    another writer created there in the meantime is never replaced.
    """
    if os.name == "nt":
        _retrying(os.rename, src, dest)  # MoveFileEx without REPLACE_EXISTING
        return
    try:
        os.link(src, dest)
    except FileExistsError:
        raise
    except OSError:
        # No hard links on this filesystem (FAT, some network shares). The write lock
        # keeps other sift writers out; only an outside writer could race this.
        if os.path.lexists(dest):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dest)) from None
        os.rename(src, dest)
        return
    with contextlib.suppress(FileNotFoundError):
        os.unlink(src)


def write_bytes_atomic(path: Path, data: bytes, *, durable: bool = False) -> None:
    """Replace `path` with `data` atomically (temp file in the same directory, then
    os.replace). For small state files - ingest state, caches - as much as notes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _write_temp(path.parent, path.stem, data, durable=durable)
    try:
        _replace_into(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    write_bytes_atomic(path, text.encode(encoding))


def _clean(value: Any) -> Any:
    """`value` with lone surrogates replaced by '?' (strings, recursively)."""
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return value.encode("utf-8", "replace").decode("utf-8")
        return value
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clean(v) for v in value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def _sanitize(note: Note) -> None:
    """Replace lone surrogates in place, before rendering.

    They arrive from scraped HTML and bad JSON, and one of them used to abort
    `write_text` after it had already truncated the file. Cleaning the note object
    itself (not just the bytes written) keeps the disk and the text later embedded
    and indexed identical.
    """
    note.body = _clean(note.body)
    meta = note.meta
    for name in type(meta).model_fields:
        value = getattr(meta, name)
        cleaned = _clean(value)
        if cleaned != value:
            setattr(meta, name, cleaned)
    extra = meta.model_extra
    if extra:
        for key in list(extra):
            extra[key] = _clean(extra[key])


# --------------------------------------------------------------------------------
# Locks and private state
# --------------------------------------------------------------------------------

_THREAD_LOCK = threading.RLock()
_FILE_LOCKS: dict[str, Any] = {}
_LOCK_TIMEOUT_S = 60.0
_lock_warned = False


def state_dir() -> Path | None:
    """sift's private state directory, ``<db dir>/_sift``.

    Caches, locks and ledgers live here and never inside the vault, where Obsidian
    would sync and display them and every walker would have to know to skip them.
    LanceDB only treats ``*.lance`` entries of the DB dir as tables. None when there
    is no usable DB dir; callers then run without persisted state.
    """
    try:
        from sift.config import get_settings

        d = get_settings().resolved_db() / "_sift"
        d.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - no writable DB dir: degrade, never fail
        log.debug("no sift state dir: %s", exc)
        return None
    return d


def vault_key(vault: Path) -> str:
    """A short stable key for a vault path (names its lock and catalog files)."""
    norm = os.path.normcase(str(Path(vault).resolve()))
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _file_lock(vault: Path) -> Any:
    d = state_dir()
    if d is None:
        return None
    lock_path = d / f"vault-{vault_key(vault)}.lock"
    key = os.path.normcase(str(lock_path))
    lock = _FILE_LOCKS.get(key)
    if lock is None:
        from filelock import FileLock

        lock = FileLock(str(lock_path), timeout=_LOCK_TIMEOUT_S)
        _FILE_LOCKS[key] = lock
    return lock


@contextlib.contextmanager
def write_lock(vault: Path | str) -> Iterator[None]:
    """Serialise vault writes: one thread at a time in this process, one process at a
    time across the CLI and the MCP server.

    Reentrant. Hold it around a whole read-modify-write (load a note, change it, save
    it) so two callers cannot interleave and lose one update; `write_note` and
    `delete_note` take it themselves.
    """
    global _lock_warned
    with _THREAD_LOCK:
        lock = _file_lock(Path(vault))
        acquired = False
        if lock is not None:
            from filelock import Timeout

            try:
                lock.acquire()
                acquired = True
            except Timeout as exc:
                raise RuntimeError(
                    f"the vault is locked by another sift process ({lock.lock_file}); "
                    f"gave up after {_LOCK_TIMEOUT_S:.0f}s"
                ) from exc
            except OSError as exc:
                # Read-only or vanished DB dir: other processes are not excluded, but
                # threads still are, and refusing every write would be worse.
                if not _lock_warned:
                    _lock_warned = True
                    log.warning("vault write lock unavailable, continuing without it: %s", exc)
        try:
            yield
        finally:
            if acquired:
                lock.release()


# --------------------------------------------------------------------------------
# Reading and parsing
# --------------------------------------------------------------------------------


def read_note_text(path: Path) -> str:
    """A note file's text. A UTF-8 BOM is tolerated: PowerShell 5.1's `Out-File` and
    `Set-Content -Encoding utf8` write one, and it hides the frontmatter fence."""
    return Path(path).read_text(encoding="utf-8-sig")


def _read_meta(path: Path) -> dict[str, Any] | None:
    """A file's frontmatter, or None when it cannot be read or parsed."""
    try:
        text = read_note_text(path)
    except (OSError, ValueError):
        return None
    try:
        metadata = frontmatter.loads(text).metadata
    except Exception:  # noqa: BLE001 - unparseable: not a note we may touch
        return None
    return metadata if isinstance(metadata, dict) else None


def _validate_lenient(metadata: Mapping[Any, Any]) -> Frontmatter:
    """Validate frontmatter, keeping bad *optional* values instead of failing.

    A missing or invalid id / type / title still fails the note: that is not a note
    sift can identify. Anything else that does not validate (``bounty: lots``,
    ``extra: [1]``) and any non-string YAML key is set aside verbatim and written back
    unchanged on save - dropping it would erase the user's value the next time
    `resolve_idea`, `ingest epss` or a distill run re-saved the note.
    """
    from pydantic import ValidationError

    data: dict[str, Any] = {}
    raw: dict[Any, Any] = {}
    for key, value in metadata.items():
        if isinstance(key, str):
            data[key] = value
        else:
            raw[key] = value
    try:
        meta = Frontmatter.model_validate(data)
    except ValidationError as exc:
        bad: list[str] = []
        for err in exc.errors(include_input=False, include_url=False):
            loc = err.get("loc") or ()
            if not loc or not isinstance(loc[0], str) or loc[0] in REQUIRED_KEYS:
                raise
            bad.append(loc[0])
        for key in dict.fromkeys(bad):
            if key in data:
                raw[key] = data.pop(key)
        meta = Frontmatter.model_validate(data)  # a second failure is a real one
    if raw:
        meta.keep_invalid(raw)
    return meta


def parse_note_text(text: str) -> tuple[Frontmatter, str]:
    """(frontmatter, body) of a note's text. Pure: no I/O.

    Raises `NotANote` for an empty file or one without a frontmatter block, a yaml
    error for unparseable frontmatter, and pydantic's ValidationError when id, type
    or title are missing or invalid. Invalid optional values do not fail the note;
    they are kept verbatim (`Frontmatter.invalid_fields`).
    """
    text = text.removeprefix("﻿")
    post = frontmatter.loads(text)
    if not post.metadata:
        raise NotANote("empty file" if not text.strip() else "no frontmatter")
    return _validate_lenient(post.metadata), post.content


def load_note(path: Path) -> Note:
    path = Path(path)
    # Stat BEFORE reading: if the file changes mid-read, the recorded mtime is older
    # than the file and the next reindex re-embeds it - the safe direction.
    mtime = path.stat().st_mtime
    meta, body = parse_note_text(read_note_text(path))
    if meta.invalid_fields:
        _report_once(
            path,
            "kept",
            "note %s: frontmatter value(s) kept verbatim but not used (invalid): %s",
            ", ".join(map(str, meta.invalid_fields)),
        )
    return Note(meta=meta, body=body, path=path, mtime=mtime)


def describe_error(exc: BaseException) -> str:
    """One line about why a file is not a usable note. Never echoes its content: a
    pydantic or YAML message would otherwise quote frontmatter values."""
    from pydantic import ValidationError

    if isinstance(exc, NotANote):
        msg = str(exc)
        if msg == "no frontmatter":
            return f"{msg} (`sift ingest local` adopts hand-written notes)"
        return msg
    if isinstance(exc, ValidationError):
        errs = exc.errors(include_input=False, include_url=False)
        parts = [
            f"{'.'.join(map(str, e.get('loc', ()))) or '<root>'}: {e.get('msg', '')}"
            for e in errs[:3]
        ]
        more = f" (+{len(errs) - 3} more)" if len(errs) > 3 else ""
        return "invalid frontmatter: " + "; ".join(parts) + more
    if isinstance(exc, yaml.MarkedYAMLError):
        mark = exc.problem_mark
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        return f"unparseable frontmatter ({type(exc).__name__}: {exc.problem or 'syntax error'}{where})"
    if isinstance(exc, yaml.YAMLError):
        return f"unparseable frontmatter ({type(exc).__name__})"
    if isinstance(exc, UnicodeDecodeError):
        return f"not UTF-8 text (bad byte at offset {exc.start})"
    first = (str(exc).splitlines() or [""])[0][:160]
    return f"{type(exc).__name__}: {first}" if first else type(exc).__name__


_REPORTED: dict[tuple[str, str], tuple[int, int]] = {}
_REPORTED_LOCK = threading.Lock()


def _report_once(
    path: Path, kind: str, template: str, *args: object, vault: Path | None = None
) -> None:
    """Log `template` once per file state per process.

    `list_notes`, `stats` and the link graph rescan the vault on every call; without
    this, one bad file repeats its warning thousands of times in a session. A file
    that changes and is still bad is reported again.
    """
    try:
        st = path.stat()
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        sig = (-1, -1)
    key = (kind, os.path.normcase(str(path)))
    with _REPORTED_LOCK:
        if _REPORTED.get(key) == sig:
            return
        _REPORTED[key] = sig
    shown: Path | str = path
    if vault is not None:
        with contextlib.suppress(ValueError):
            shown = path.relative_to(vault)
    log.warning(template, shown, *args)


def report_unreadable(path: Path, exc: BaseException, *, vault: Path | None = None) -> None:
    """Warn - once per file state per process - that `path` is skipped, and why."""
    _report_once(path, "skip", "skipping unreadable note %s: %s", describe_error(exc), vault=vault)


# --------------------------------------------------------------------------------
# Walking the vault
# --------------------------------------------------------------------------------

_SKIP_FILES = frozenset({"readme.md"})


@dataclass(frozen=True)
class _IgnoreRules:
    names: frozenset[str]  # casefolded folder names, skipped at any depth
    paths: frozenset[str]  # casefolded vault-relative folder paths


def _ignore_rules(vault: Path) -> _IgnoreRules:
    """Folders that hold no notes beyond the structural rules: the
    `vault_ignore_dirs` setting (a name matches at any depth, a path with a slash is
    vault-relative) and the folder Obsidian's core Templates plugin uses - template
    placeholders are not notes."""
    names: set[str] = set()
    paths: set[str] = set()
    try:
        from sift.config import get_settings

        configured = str(getattr(get_settings(), "vault_ignore_dirs", "") or "")
    except Exception:  # noqa: BLE001 - settings unavailable: structural rules only
        configured = ""
    for entry in configured.split(","):
        entry = entry.strip().replace("\\", "/").strip("/").casefold()
        if entry:
            (paths if "/" in entry else names).add(entry)
    try:
        cfg = json.loads((vault / ".obsidian" / "templates.json").read_text(encoding="utf-8"))
        folder = cfg.get("folder") if isinstance(cfg, dict) else None
        if isinstance(folder, str) and folder.strip().strip("/\\"):
            paths.add(folder.strip().replace("\\", "/").strip("/").casefold())
    except (OSError, ValueError):
        pass
    return _IgnoreRules(frozenset(names), frozenset(paths))


def _ignored_dir(path: str, name: str, vault: Path, rules: _IgnoreRules) -> bool:
    if name.startswith((".", "_")):  # .obsidian, .trash, .git, _templates
        return True
    if name.casefold() in rules.names:
        return True
    if rules.paths:
        try:
            rel = Path(path).relative_to(vault).as_posix().casefold()
        except ValueError:
            return False
        return rel in rules.paths
    return False


def _walk_dir(
    directory: Path,
    vault: Path,
    rules: _IgnoreRules,
    on_dir: Callable[[Path], None] | None,
) -> Iterator[tuple[Path, os.stat_result]]:
    if on_dir is not None:
        on_dir(directory)  # before listing: a change after this point is never absorbed
    try:
        with os.scandir(directory) as it:
            entries = sorted(it, key=lambda e: os.path.normcase(e.name))
    except OSError:
        return
    for entry in entries:
        name = entry.name
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if is_dir:
            if not _ignored_dir(entry.path, name, vault, rules):
                yield from _walk_dir(Path(entry.path), vault, rules, on_dir)
            continue
        if (
            name.startswith((".", "_"))
            or not name.lower().endswith(".md")
            or name.casefold() in _SKIP_FILES
        ):
            continue
        try:
            st = entry.stat()
        except OSError:
            continue
        if _stat.S_ISREG(st.st_mode):
            yield Path(entry.path), st


def walk_note_entries(
    root: Path,
    *,
    vault: Path | None = None,
    on_dir: Callable[[Path], None] | None = None,
) -> Iterator[tuple[Path, os.stat_result]]:
    """Every candidate note file under `root`, in path order, with its stat.

    The one enumeration rule, for every vault walker: `.md` files (any case), not
    `_`- or dot-prefixed, not README.md (any case), outside dot- and `_`-folders
    (`.obsidian`, `.trash`, `_templates`) and outside ignored folders (the
    `vault_ignore_dirs` setting, and Obsidian's Templates folder). Order matches
    `sorted(root.rglob("*.md"))`. Content checks - empty, no frontmatter, invalid -
    are the caller's. `on_dir` is called with each folder walked, before it is
    listed (the catalog records folder mtimes with it).
    """
    root = Path(root)
    vault = Path(vault) if vault is not None else root
    yield from _walk_dir(root, vault, _ignore_rules(vault), on_dir)


def walk_note_files(root: Path, *, vault: Path | None = None) -> Iterator[Path]:
    """`walk_note_entries` without the stat."""
    for path, _st in walk_note_entries(root, vault=vault):
        yield path


def iter_notes(vault: Path, *, note_type: str | None = None) -> Iterator[Note]:
    """Every readable note under `vault` (or `vault/<note_type>`), in path order.

    Fresh objects from disk on every call, so callers may mutate and save them.
    Files that are empty, have no frontmatter or do not validate are skipped with
    one warning per file per process - logged, never printed. A second file carrying
    an id already yielded in this pass is still yielded (nothing is dropped on an id
    alone) but reported once, so a duplicate surfaces instead of silently making
    the index alternate between the copies.
    """
    vault = Path(vault)
    root = vault / note_type if note_type else vault
    if not root.is_dir():
        return
    first_path: dict[str, Path] = {}
    for path, _st in walk_note_entries(root, vault=vault):
        try:
            note = load_note(path)
        except FileNotFoundError:
            continue  # removed between the directory listing and the read
        except Exception as exc:  # noqa: BLE001 - one bad file never stops a scan
            report_unreadable(path, exc, vault=vault)
            continue
        earlier = first_path.setdefault(note.meta.id, path)
        if earlier != path:
            _report_once(
                path,
                "dup",
                "note %s carries id %s, which %s also carries (duplicate files for one id)",
                note.meta.id,
                _rel(earlier, vault),
                vault=vault,
            )
        yield note


def _rel(path: Path, vault: Path) -> str:
    try:
        return path.relative_to(vault).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------
# Finding the file that carries an id
# --------------------------------------------------------------------------------

# How stale the catalog may be when the write path looks an id up. A stat walk of the
# real vault costs tens of ms; at most one a second keeps a bulk ingest from paying it
# per note. Within that window the catalog still re-walks as soon as any walked
# folder's mtime changed (a create, rename or delete by Obsidian or another process),
# and a stale *hit* is always re-verified against the file. Only an in-place edit that
# gives an existing file a new id goes unseen for that second - and then the title
# path is still checked on disk.
_CARRIER_MAX_AGE_S = 1.0


@dataclass
class _Carrier:
    path: Path
    meta: dict[str, Any]
    trusted: bool  # handed in by the caller (note.path / existing), not found by id


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _verified_meta(path: Path | None, note_id: str) -> dict[str, Any] | None:
    """The frontmatter at `path` if that file exists and carries `note_id`."""
    if path is None:
        return None
    path = Path(path)
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    metadata = _read_meta(path)
    return metadata if _id_of(metadata) == note_id else None


def _verify_rows(rows: Any, note_id: str) -> tuple[list[tuple[Path, dict[str, Any]]], list[Path]]:
    found: list[tuple[Path, dict[str, Any]]] = []
    stale: list[Path] = []
    for row in rows:
        metadata = _verified_meta(row.path, note_id)
        if metadata is None:
            stale.append(row.path)
        else:
            found.append((row.path, metadata))
    return found, stale


def _find_carriers(
    vault: Path, note_id: str, *, max_age: float
) -> list[tuple[Path, dict[str, Any]]]:
    """Every file carrying `note_id`, each verified against its file on disk.

    The catalog is advisory. A row that disagrees with its file (renamed, deleted or
    re-ided since) makes the disagreeing files re-read and the vault re-walked, which
    also finds a renamed copy under its new name; only verified files are returned.
    """
    from sift.vault.catalog import get_catalog

    cat = get_catalog(vault)
    cat.ensure_fresh(max_age=max_age)
    found, stale = _verify_rows(cat.by_id(note_id), note_id)
    if stale:
        cat.invalidate(stale)
        cat.refresh()
        found, _stale = _verify_rows(cat.by_id(note_id), note_id)
    return found


def _same_path(a: Path, b: Path) -> bool:
    if os.path.normcase(str(a)) == os.path.normcase(str(b)):
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _numbered_rx(stem: str, suffix: str) -> re.Pattern[str]:
    flags = re.IGNORECASE if os.name == "nt" else 0
    return re.compile(re.escape(stem) + r" \((\d+)\)" + re.escape(suffix), flags)


def _named_after(stem: str, title_stem: str) -> bool:
    """`stem` is `title_stem` or `title_stem (n)` - the names sift gives a title."""
    if os.path.normcase(stem) == os.path.normcase(title_stem):
        return True
    return _numbered_rx(title_stem, "").fullmatch(stem) is not None


def _rank(
    found: list[tuple[Path, dict[str, Any]]], note: Note, vault: Path
) -> list[tuple[Path, dict[str, Any]]]:
    """Best carrier first: the same document, then the one at its title path, then
    the newest."""
    title_path = note_path(vault, note.meta)

    def key(item: tuple[Path, dict[str, Any]]) -> tuple[bool, bool, int, str]:
        path, metadata = item
        same = same_document(metadata, note.meta)[0]
        at_title = os.path.normcase(str(path.parent)) == os.path.normcase(
            str(title_path.parent)
        ) and _named_after(path.stem, title_path.stem)
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = 0
        return (not same, not at_title, -mtime, os.path.normcase(str(path)))

    return sorted(found, key=key)


def _locate(
    vault: Path, note: Note, *, existing: Path | None, locate_by_id: bool
) -> tuple[_Carrier | None, list[Path]]:
    note_id = note.meta.id
    for explicit in (existing, note.path):
        if explicit is None:
            continue
        metadata = _verified_meta(explicit, note_id)
        if metadata is not None:
            return _Carrier(Path(explicit), metadata, trusted=True), []
    if not locate_by_id:
        return None, []
    found = _find_carriers(vault, note_id, max_age=_CARRIER_MAX_AGE_S)
    if not found:
        return None, []
    ranked = _rank(found, note, vault)
    best_path, best_meta = ranked[0]
    return _Carrier(best_path, best_meta, trusted=False), [p for p, _ in ranked[1:]]


def locate_note(
    vault: Path,
    note_id: str,
    *,
    meta: Frontmatter | None = None,
    max_age: float = _CARRIER_MAX_AGE_S,
) -> Path | None:
    """The file that carries `note_id` (verified on disk), or None.

    With `meta`, only a file holding the same document counts (see `same_document`):
    the check an ingest source wants before skipping a fetch as "already have it".
    `max_age` bounds how stale the catalog may be (0 = re-stat the vault first); the
    default keeps a per-entry check in a bulk ingest from walking the vault each time.
    """
    found = _find_carriers(Path(vault), note_id, max_age=max_age)
    if meta is not None:
        found = [(p, md) for p, md in found if same_document(md, meta)[0]]
        if found:
            found = _rank(found, Note(meta=meta, body=""), Path(vault))
    return found[0][0] if found else None


# --------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------


def _resolve_free(dest: Path, note_id: str) -> tuple[Path, dict[str, Any] | None]:
    """Where a note with `note_id` goes for the title-derived `dest`, plus the
    frontmatter there when that file already holds `note_id`.

    Since the filename is the title, two notes with the same title want the same
    file - and silently overwriting one was a real data-loss bug. Reuse a path that
    already holds *this* id (`dest` or any `dest (n)` sibling, so a gap in the chain
    can never fork a second copy), otherwise take the first free `Title (n).md`,
    Obsidian's own convention. Unreadable files count as occupied. Always reads the
    disk, never the catalog.
    """
    if not _exists(dest):
        return dest, None
    metadata = _read_meta(dest)
    if _id_of(metadata) == note_id:
        return dest, metadata
    rx = _numbered_rx(dest.stem, dest.suffix)
    numbered: dict[int, Path] = {}
    with contextlib.suppress(OSError):
        for cand in dest.parent.glob(f"{glob.escape(dest.stem)} (*){dest.suffix}"):
            m = rx.fullmatch(cand.name)
            if m:
                numbered[int(m.group(1))] = cand
    for n in sorted(numbered):
        metadata = _read_meta(numbered[n])
        if _id_of(metadata) == note_id:
            return numbered[n], metadata
    for n in range(2, 1000):
        cand = dest.with_name(f"{dest.stem} ({n}){dest.suffix}")
        if not _exists(cand):
            return cand, None
    raise RuntimeError(f"could not find a free filename for {dest}")


def _free_path(dest: Path, note_id: str) -> Path:
    """`dest`, the `dest (n)` sibling already holding `note_id`, or the first free one."""
    return _resolve_free(dest, note_id)[0]


def _check_identity(path: Path, on_disk: Mapping[str, Any], meta: Frontmatter) -> None:
    same, reason = same_document(on_disk, meta)
    if not same:
        source = on_disk.get("source")
        url = on_disk.get("url")
        raise IdConflict(
            meta.id,
            path,
            reason,
            existing_source=None if source is None else str(source),
            existing_url=None if url is None else str(url),
        )


def _rewrite(path: Path, data: bytes, *, durable: bool) -> None:
    write_bytes_atomic(path, data, durable=durable)


def _follow_title(carrier: _Carrier, note: Note) -> Path:
    """Rename the rewritten carrier when its title-derived filename changed.

    Only when sift named the file after its old title (`Old.md` / `Old (n).md`); a
    filename the user chose is theirs, and inbound `[[links]]` resolve by it. The new
    name is taken within the same folder, through `_resolve_free`, and the move is
    no-clobber, so it can neither land on another note nor leave two files for one
    id. If the rename cannot happen the note keeps its old name: content first.
    """
    current = carrier.path
    old_title = carrier.meta.get("title")
    if old_title is None:
        return current
    note_id = note.meta.id
    new_stem = title_to_filename(note.meta.title, note_id)
    old_stem = title_to_filename(str(old_title), note_id)
    if os.path.normcase(new_stem) == os.path.normcase(old_stem):
        return current
    if not _named_after(current.stem, old_stem):
        return current
    target = current.with_name(new_stem + current.suffix)
    for _ in range(_CLAIM_ATTEMPTS):
        dest, holder = _resolve_free(target, note_id)
        if _same_path(dest, current):
            return current
        if holder is not None:
            log.warning(
                "kept %s: %s already carries id %s (duplicate files for one id)",
                current.name,
                dest.name,
                note_id,
            )
            return current
        try:
            _move_no_clobber(current, dest)
        except FileExistsError:
            continue  # taken between the check and the move: look again
        except OSError as exc:
            log.warning(
                "kept %s under its old name: renaming to %s failed: %s",
                current.name,
                dest.name,
                describe_error(exc),
            )
            return current
        return dest
    return current


def _create(
    vault: Path, note: Note, data: bytes, *, durable: bool, verify_identity: bool
) -> SaveResult:
    target = note_path(vault, note.meta)
    target.parent.mkdir(parents=True, exist_ok=True)
    note_id = note.meta.id
    tmp = _write_temp(target.parent, target.stem, data, durable=durable)
    try:
        for _ in range(_CLAIM_ATTEMPTS):
            dest, holder = _resolve_free(target, note_id)
            if holder is not None:
                # The title path (or a numbered sibling) holds this id although the
                # lookup did not report it - lookup disabled, or a file another
                # process created moments ago. It is an existing note after all.
                if verify_identity:
                    _check_identity(dest, holder, note.meta)
                _replace_into(tmp, dest)
                return SaveResult(dest, created=False, previous_path=dest, title_clash=False)
            try:
                _move_no_clobber(tmp, dest)
            except FileExistsError:
                continue  # another writer claimed the name between the check and the move
            return SaveResult(dest, created=True, previous_path=None, title_clash=dest != target)
        raise RuntimeError(f"could not claim a filename for {target}")
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


_DUPLICATES_WARNED: set[str] = set()


def _warn_duplicates(note_id: str, wrote: Path, others: list[Path], vault: Path) -> None:
    if note_id in _DUPLICATES_WARNED:
        return
    _DUPLICATES_WARNED.add(note_id)
    log.warning(
        "id %s is carried by %d files; wrote %s, left %s untouched",
        note_id,
        len(others) + 1,
        _rel(wrote, vault),
        ", ".join(_rel(p, vault) for p in others),
    )


def _catalog_saved(vault: Path, note: Note, previous: Path | None) -> None:
    try:
        from sift.vault.catalog import peek_catalog

        cat = peek_catalog(vault)
        if cat is not None:
            cat.note_saved(note, previous=previous)
    except Exception as exc:  # noqa: BLE001 - the catalog is advisory
        log.debug("catalog update after saving %s failed: %s", note.path, exc)


def _catalog_removed(vault: Path, path: Path) -> None:
    try:
        from sift.vault.catalog import peek_catalog

        cat = peek_catalog(vault)
        if cat is not None:
            cat.note_removed(path)
    except Exception as exc:  # noqa: BLE001 - the catalog is advisory
        log.debug("catalog update after removing %s failed: %s", path, exc)


def write_note(
    vault: Path,
    note: Note,
    *,
    stamp: bool = True,
    existing: Path | None = None,
    locate_by_id: bool = True,
    verify_identity: bool = True,
    rename: bool = True,
    durable: bool = False,
) -> SaveResult:
    """Save `note`, as an upsert by `meta.id`, atomically. See the module docstring.

    Where it goes:

    1. `existing`, then `note.path`, when that file still carries this id - the file
       a loaded note came from, even if the user renamed or moved it since. Trusted:
       no identity check.
    2. Otherwise (unless `locate_by_id=False`) the file that carries the id, found
       through the catalog and verified on disk. If it holds a different document
       (`same_document`) nothing is written and `IdConflict` is raised; pass
       `verify_identity=False` only when you have decided it is the same note (a
       merge). With several files for one id, the matching one at its title path
       wins and the rest are reported in `SaveResult.duplicates`, never touched.
    3. Otherwise a new file at `note_path()`, or `Title (n).md` if that name belongs
       to another note (`SaveResult.title_clash`).

    A rewritten file is renamed when its title-derived name changed (`rename=False`
    keeps it). `durable=True` adds an fsync - power-loss safety, at a cost in bulk.
    """
    vault = Path(vault)
    if not note.meta.id:
        raise ValueError("cannot save a note without an id")
    if stamp and note.meta.ingested is None:
        note.meta.ingested = datetime.now(UTC)
    _sanitize(note)
    data = note.render().encode("utf-8", "replace")

    with write_lock(vault):
        carrier, others = _locate(vault, note, existing=existing, locate_by_id=locate_by_id)
        if carrier is not None:
            if verify_identity and not carrier.trusted:
                _check_identity(carrier.path, carrier.meta, note.meta)
            if others:
                _warn_duplicates(note.meta.id, carrier.path, others, vault)
            _rewrite(carrier.path, data, durable=durable)
            final = _follow_title(carrier, note) if rename else carrier.path
            result = SaveResult(
                final,
                created=False,
                previous_path=carrier.path,
                title_clash=False,
                duplicates=tuple(others),
            )
        else:
            result = _create(vault, note, data, durable=durable, verify_identity=verify_identity)
        note.path = result.path
        try:
            note.mtime = result.path.stat().st_mtime
        except OSError:
            note.mtime = None
        _catalog_saved(vault, note, result.previous_path)
    return result


def save_note(
    vault: Path,
    note: Note,
    *,
    stamp: bool = True,
    existing: Path | None = None,
    locate_by_id: bool = True,
    verify_identity: bool = True,
    rename: bool = True,
    durable: bool = False,
) -> Path:
    """`write_note`, returning only where the note now lives."""
    return write_note(
        vault,
        note,
        stamp=stamp,
        existing=existing,
        locate_by_id=locate_by_id,
        verify_identity=verify_identity,
        rename=rename,
        durable=durable,
    ).path


# --------------------------------------------------------------------------------
# Deleting
# --------------------------------------------------------------------------------


def _trash_destination(vault: Path, path: Path) -> Path:
    try:
        rel = path.relative_to(vault)
    except ValueError:
        rel = Path(path.name)
    target = vault / TRASH_DIR / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _stamp_deleted(path: Path, reason: str) -> None:
    try:
        trashed = load_note(path)
        trashed.meta.extra["deleted"] = datetime.now(UTC).isoformat(timespec="seconds")
        trashed.meta.extra["deleted_reason"] = reason
        _sanitize(trashed)
        _rewrite(path, trashed.render().encode("utf-8", "replace"), durable=False)
    except Exception as exc:  # noqa: BLE001 - the move already happened; the stamp is a courtesy
        log.warning("trashed %s but could not record why: %s", path.name, describe_error(exc))


def delete_note(
    note_id: str, *, vault: Path | None = None, reason: str | None = None
) -> list[Path]:
    """Soft-delete: move every file carrying `note_id` into ``<vault>/.trash/``.

    Nothing is unlinked - a wrong delete is one move away from undone - and `.trash`
    is skipped by every vault walker, so the note leaves listings, lookups and the
    next reindex. The folder layout is kept (`.trash/<type>/<name>.md`, with ` (n)` on
    a clash), and `reason` is stamped into the trashed copy's `extra`. Returns the new
    locations; empty when no file carries the id. Removing the id from the index
    (`Store.delete_note`) is the caller's job.
    """
    if vault is None:
        from sift.config import get_settings

        vault = get_settings().resolved_vault()
    vault = Path(vault)
    moved: list[Path] = []
    with write_lock(vault):
        for path, _metadata in _find_carriers(vault, note_id, max_age=0.0):
            target = _trash_destination(vault, path)
            for n in range(1, _CLAIM_ATTEMPTS):
                dest = target if n == 1 else target.with_name(f"{target.stem} ({n}){target.suffix}")
                if _exists(dest):
                    continue
                try:
                    _move_no_clobber(path, dest)
                except FileExistsError:
                    continue
                break
            else:
                raise RuntimeError(f"could not find a free name in {target.parent}")
            moved.append(dest)
            _catalog_removed(vault, path)
            if reason:
                _stamp_deleted(dest, reason)
    return moved
