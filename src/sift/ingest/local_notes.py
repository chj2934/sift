"""Index hand-authored markdown and backfill missing frontmatter.

Drop a `.md` file anywhere under the vault (e.g. `vault/inbox/idea.md`). If it has no
frontmatter, or lacks one of ``id``, ``type`` and ``title``, we fill in only what is
missing - ``type`` from its parent folder (default `finding`), ``title`` from the first
heading or the filename, a stable ``id`` - and rewrite it as a proper note before
indexing.

What the backfill must never do, because each of these happened to real notes:

* **Rebuild the frontmatter.** It used to keep five fields and drop the rest: the
  report URL, program, severity, date and links of the user's own findings, the most
  valuable notes in the vault. Every existing key is kept now, unknown ones included.
* **Split a scalar tag into characters** (``tags: ssrf`` became s, s, r, f).
* **Mangle a UTF-8-BOM note** (PowerShell writes one): the BOM hid the frontmatter,
  the note got a new id, its YAML was demoted into the body and the original deleted.
* **Coerce a complete note's metadata.** A note with id, type and title is left as it
  is even when a value is invalid (``type: idea``); it is reported instead.
* **Abort the run on one bad file.** Each file that cannot be backfilled is left
  untouched and reported.

Only the files the vault catalog could not load are even read (everything else is
already a valid note), and indexing is the incremental `pipeline.reindex`: unchanged
notes are skipped by mtime and changed ones embedded in batches. The old per-note loop
re-embedded all ~13.9k notes with two index commits each on every run.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import frontmatter
from slugify import slugify

from sift.config import get_settings
from sift.ingest.existing import same_file
from sift.vault.notes import (
    NotANote,
    Note,
    describe_error,
    load_note,
    locate_note,
    note_path,
    read_note_text,
    save_note,
    write_lock,
    write_text_atomic,
)
from sift.vault.schema import NOTE_TYPES, Frontmatter

log = logging.getLogger(__name__)

_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _infer_type(path: Path, vault: Path) -> str:
    for part in path.relative_to(vault).parts[:-1]:
        if part in NOTE_TYPES:
            return part
    return "finding"


def _needs_backfill(meta: dict) -> bool:
    return not all(k in meta and meta[k] for k in ("id", "type", "title"))


@dataclass
class BackfillResult:
    """What `backfill_and_index` did. Unpacks as ``fixed, indexed`` like the old
    2-tuple, so existing callers keep working."""

    fixed: int = 0  # files given complete frontmatter
    indexed: int = 0  # notes (re-)embedded by the incremental reindex
    problems: list[tuple[Path, str]] = field(default_factory=list)  # left untouched, with why
    stats: Any = None  # the `pipeline.ReindexStats` of the indexing pass

    def __iter__(self) -> Iterator[int]:
        return iter((self.fixed, self.indexed))


def _local_id(vault: Path, path: Path) -> str:
    """``local-<stem>``, unless another file already carries that id (two `idea.md` in
    different folders): then a short hash of this file's vault path is appended, so
    neither note can overwrite the other."""
    base = f"local-{slugify(path.stem, max_length=60)}".rstrip("-")
    rel = path.relative_to(vault).as_posix()
    if base == "local":
        return f"local-{hashlib.sha1(rel.encode('utf-8')).hexdigest()[:8]}"
    holder = locate_note(vault, base)
    if holder is None or same_file(holder, path):
        return base
    return f"{base}-{hashlib.sha1(rel.encode('utf-8')).hexdigest()[:8]}"


def build_backfilled(vault: Path, path: Path, text: str) -> Note | None:
    """The note `text` (the file at `path`) becomes, or None when its frontmatter is
    already complete (a complete-but-invalid note is reported, never coerced).

    Fills only missing or empty ``id``/``type``/``title``/``source``/``ingested``;
    every other key - modelled or not - is kept as written. Raises (and the file is
    left untouched) when the result would not validate.
    """
    if not text.strip():
        raise NotANote("empty file")
    post = frontmatter.loads(text)
    data: dict[str, Any] = dict(post.metadata)
    if not _needs_backfill(data):
        return None
    body = (post.content if post.metadata else text).strip()

    extra = data.get("extra")
    if extra is None:
        extra = {}
    elif not isinstance(extra, dict):
        raise ValueError("`extra` is not a mapping; fix it by hand")
    else:
        extra = dict(extra)

    if not data.get("title"):
        m = _H1.search(body)
        data["title"] = m.group(1).strip() if m else path.stem.replace("-", " ").title()
    raw_type = data.get("type")
    wanted = str(raw_type).strip().lower() if raw_type else ""
    if wanted not in NOTE_TYPES:
        if wanted:
            extra.setdefault("original_type", raw_type)  # keep what the user wrote
        data["type"] = _infer_type(path, vault)
    if not data.get("id"):
        data["id"] = _local_id(vault, path)
    if not data.get("source"):
        data["source"] = "manual"
    if not data.get("ingested"):
        data["ingested"] = datetime.now(UTC)
    if extra or "extra" in data:
        data["extra"] = extra

    meta = Frontmatter.model_validate(data)  # a bad optional value: report, don't guess
    return Note(meta=meta, body=body)


def _unlink(path: Path, attempts: int = 6) -> None:
    """Remove the loose original, retrying a Windows sharing violation briefly."""
    for attempt in range(attempts):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def backfill_one(vault: Path, path: Path) -> Path | None:
    """Backfill one file; returns where the note now lives, or None when it did not
    need backfilling. Raises with the file untouched when it cannot be done."""
    with write_lock(vault):
        note = build_backfilled(vault, path, read_note_text(path))
        if note is None:
            return None
        dest = note_path(vault, note.meta)
        if dest.exists() and same_file(dest, path):
            # The file already sits at its canonical name (case-insensitively, on
            # Windows): rewrite it in place. Saving "elsewhere" would see this very
            # file as occupied and fork `Title (2).md`.
            write_text_atomic(path, note.render())
            return path
        # Loose files move to their canonical place; never look the (new) id up.
        new_path = save_note(vault, note, stamp=False, locate_by_id=False)
        if same_file(new_path, path):
            return new_path
        if load_note(new_path).meta.id != note.meta.id:
            raise RuntimeError(f"backfilled copy {new_path.name} did not verify; original kept")
        try:
            _unlink(path)
        except OSError as exc:
            raise RuntimeError(
                f"backfilled to {new_path.name}, but the original could not be removed "
                f"({describe_error(exc)}); delete it by hand"
            ) from exc
        return new_path


def backfill(vault: Path | None = None) -> tuple[int, list[tuple[Path, str]]]:
    """Backfill every file the vault catalog could not load. Returns (fixed, problems).

    Files are found by the shared vault walker, so `.trash`, `.obsidian`, templates,
    ``_``-files, README.md and SIFT_VAULT_IGNORE_DIRS are never adopted.
    """
    from sift.vault.catalog import fresh_catalog

    vault = Path(vault) if vault is not None else get_settings().resolved_vault()
    fixed = 0
    problems: list[tuple[Path, str]] = []
    for path, why in fresh_catalog(vault).skipped():
        try:
            moved_to = backfill_one(vault, path)
        except Exception as exc:  # noqa: BLE001 - one bad file never stops the run
            problems.append((path, describe_error(exc)))
            continue
        if moved_to is None:
            problems.append((path, why))  # complete frontmatter, but not loadable
        else:
            fixed += 1
    for path, why in problems:
        log.warning("notes: left %s untouched: %s", path, why)
    return fixed, problems


def backfill_and_index(*, on_progress: Callable[..., Any] | None = None) -> BackfillResult:
    """Backfill frontmatter, then bring the index up to date incrementally.

    ``fixed, indexed = backfill_and_index()`` still works; the result also carries the
    files left untouched (``problems``) and the reindex stats.
    """
    from sift.pipeline import reindex

    vault = get_settings().resolved_vault()
    fixed, problems = backfill(vault)
    stats = reindex(vault, on_progress=on_progress)
    return BackfillResult(fixed=fixed, indexed=stats.notes, problems=problems, stats=stats)
