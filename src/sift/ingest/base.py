"""Shared plumbing for ingestion sources.

A source is anything that yields :class:`~sift.vault.notes.Note` objects. The
runner writes them to the vault, indexes them, and records a per-source
timestamp in ``vault/_state.json`` (used by ``sift status``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sift.config import get_settings
from sift.index.store import Store
from sift.pipeline import index_notes
from sift.vault.notes import Note, note_path, save_note
from sift.vault.schema import Frontmatter

STATE_FILE = "_state.json"
_CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)


def clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def have_note(vault: Path, meta: Frontmatter) -> bool:
    """True when this exact note is already on disk, by id.

    Lets a network source skip a body fetch it has already paid for. The filename
    is the title, so the path is deterministic for a given source record - but two
    unrelated notes can still want the same filename, which is why the id on disk
    is checked rather than mere existence. Getting that backwards would make a
    title collision look like "already ingested" and silently drop the second note.
    """
    p = note_path(vault, meta)
    if not p.exists():
        return False
    try:
        import frontmatter

        return frontmatter.loads(p.read_text(encoding="utf-8")).metadata.get("id") == meta.id
    except Exception:  # noqa: BLE001 - unreadable/partial file: re-ingest rather than skip
        return False


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


@dataclass
class IngestResult:
    source: str
    written: int = 0
    indexed_chunks: int = 0
    errors: int = 0
    # Slug collisions that were disambiguated rather than silently overwritten.
    collisions: int = 0


def _state_path() -> Path:
    return get_settings().resolved_vault() / STATE_FILE


def load_state() -> dict:
    p = _state_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def record_run(source: str, result: IngestResult) -> None:
    p = _state_path()
    state = load_state()
    state[source] = {
        "last_run": datetime.now(UTC).isoformat(timespec="seconds"),
        "written": result.written,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_source(
    source_name: str,
    notes: Iterable[Note],
    *,
    reindex_fts: bool = True,
    flush_every: int = 200,
    on_progress: Callable[[int, str], None] | None = None,
) -> IngestResult:
    """Save each note to the vault, then index in batches of ``flush_every``.

    `on_progress` receives (notes written so far, latest title). Sources are
    generators over the network, so the total is unknowable up front - the CLI shows
    a running count rather than a percentage.
    """
    s = get_settings()
    vault = s.resolved_vault()
    store = Store()
    res = IngestResult(source=source_name)
    pending: list[Note] = []

    def flush() -> None:
        if not pending:
            return
        try:
            res.indexed_chunks += index_notes(pending, store)
        except Exception as exc:  # noqa: BLE001
            res.errors += len(pending)
            print(f"  ! {source_name}: index batch failed: {exc}")
        pending.clear()
        if not on_progress:  # the caller is rendering a bar; this would tear it
            print(f"  .. {res.written} notes")

    # Filenames are note titles, so two notes sharing a title want the same file.
    # `save_note` resolves that to `Title (2).md`; this just reports how often it
    # happened, because an ingest that says "492 new notes, 0 errors" while leaving
    # 488 files on disk is how the original data-loss bug hid.
    for note in notes:
        try:
            if note.meta.ingested is None:
                note.meta.ingested = datetime.now(UTC)
            intended = note_path(vault, note.meta)
            actual = save_note(vault, note)
            if actual != intended:
                res.collisions += 1
                print(f"  ~ {source_name}: title clash, saved as {actual.name}")
            pending.append(note)
            res.written += 1
            if on_progress:
                on_progress(res.written, note.meta.title)
        except Exception as exc:  # noqa: BLE001
            res.errors += 1
            print(f"  ! {source_name}: failed on {note.meta.id}: {exc}")
        if len(pending) >= flush_every:
            flush()

    flush()
    if reindex_fts and res.written:
        store.ensure_fts()
    record_run(source_name, res)
    return res


def iter_limited(it: Iterator, limit: int | None) -> Iterator:
    if limit is None:
        yield from it
        return
    for i, x in enumerate(it):
        if i >= limit:
            return
        yield x
