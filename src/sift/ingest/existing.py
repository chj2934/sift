"""Source-side decisions about notes the vault already holds.

`base.run_source` upserts by id and skips unchanged notes, but three decisions belong
to the source because only it knows what its records mean:

* **Immutable ids** (``h1-<report>``, ``h1act-<report>``, ``chromium-fix-<sha>``) never
  describe anything new once stored. `stored_ids` lets a source skip them before it
  builds a note or reads a file, so a re-run costs nothing and never overwrites what
  the user added to the note since.
* **Merges.** KEV and NVD both key on the CVE number, and an h1-mine re-run must keep
  the user's annotations. `merge_into_existing` loads the file that carries the id,
  applies the source's merge and returns the result with ``.path`` set to that file.
  `vault.notes.write_note` trusts a note's own path when that file carries its id,
  so the merged note is written there - one file per id, whatever its title says.
* **Ids that are the identity by construction** (``chromium-doc-<path>``: the document
  at that path). `attach_existing` points the note at its file, so a sync that changes
  both the revision URL and the H1 still updates it instead of being refused as a
  different document.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path

from sift.vault.notes import IdConflict, Note, describe_error, load_note, locate_note

log = logging.getLogger(__name__)

Merge = Callable[[Note, Note], Note | None]


def stored_ids(vault: Path, *, prefix: str | None = None) -> set[str]:
    """Ids of every note in the vault (one stat walk via the catalog), optionally only
    those starting with `prefix`."""
    from sift.vault.catalog import fresh_catalog

    rows = fresh_catalog(vault).rows()
    return {r.id for r in rows if prefix is None or r.id.startswith(prefix)}


_TWINS_REPORTED: set[str] = set()


def _report_twins(vault: Path, note_id: str, merging_into: Path) -> None:
    """Say once per id when other files carry it too (KEV/NVD pairs written before
    merges existed). They are left alone - judging which body to keep is the user's
    call - but the merge only reaches one of them."""
    if note_id in _TWINS_REPORTED:
        return
    try:
        from sift.vault.catalog import get_catalog

        others = [
            r.path for r in get_catalog(vault).by_id(note_id) if not same_file(r.path, merging_into)
        ]
    except Exception:  # noqa: BLE001 - the catalog is advisory
        return
    if others:
        _TWINS_REPORTED.add(note_id)
        log.warning(
            "%s is carried by %d files; merging into %s, leaving %s as it is "
            "(merge or delete it by hand)",
            note_id,
            len(others) + 1,
            merging_into.name,
            ", ".join(p.name for p in others),
        )


def merge_into_existing(vault: Path, note: Note, merge: Merge) -> Note:
    """`note` merged into the file that already carries its id, or `note` unchanged.

    * No file carries the id: `note` as it is (a new note).
    * The merge returns a note: that note, ``path`` set to the carrier.
    * The merge returns None (nothing to change): the stored note itself, so the
      runner sees it as unchanged.
    * The file cannot be read, or the merge refuses (`IdConflict`): `note` as it is,
      and the writer's identity check decides - a different document under the id is
      refused and counted, never overwritten.

    When several files carry the id (twins written before upsert-by-id), the merge is
    computed against the twin that holds the same document as the result (what
    `locate_note(meta=...)` picks), and ``path`` names that twin. `run_source` writes
    a note to its own ``path`` when that file carries the id, so one twin's merged
    content is never written over the other twin.
    """
    note_id = note.meta.id
    path = locate_note(vault, note_id)
    if path is None:
        return note
    visited: list[Path] = []
    while True:
        try:
            stored = load_note(path)
        except Exception as exc:  # noqa: BLE001 - unreadable right now: let the writer decide
            log.warning(
                "could not read %s to merge %s into it: %s",
                path.name,
                note_id,
                describe_error(exc),
            )
            return note
        try:
            merged = merge(stored, note)
        except IdConflict as exc:
            log.debug("not merging %s: %s", note_id, exc)
            return note
        result = stored if merged is None else merged
        target = locate_note(vault, note_id, meta=result.meta)
        if target is None or same_file(target, path):
            break
        visited.append(path)
        if any(same_file(target, p) for p in visited):
            # No stable choice between twins: let the writer's identity check decide
            # rather than risk writing one twin's content into the other.
            log.warning("%s: twin files disagree on which to update; not merging", note_id)
            return note
        path = target
    _report_twins(vault, note_id, path)
    if merged is None:
        return stored
    merged.path = path
    return merged


def attach_existing(vault: Path, note: Note) -> Note:
    """Point `note` at the file already carrying its id when that file comes from the
    same source; the writer then updates that file whatever else changed."""
    path = locate_note(vault, note.meta.id)
    if path is None:
        return note
    try:
        stored = load_note(path)
    except Exception:  # noqa: BLE001 - unreadable: let the writer decide
        return note
    if (stored.meta.source or "").casefold() == (note.meta.source or "").casefold():
        note.path = path
    return note


def union(*seqs: Iterable[str]) -> list[str]:
    """Order-preserving union (first occurrence wins), skipping empty values."""
    out: dict[str, None] = {}
    for seq in seqs:
        for item in seq or ():
            if item:
                out.setdefault(item, None)
    return list(out)


def same_file(a: Path | None, b: Path | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return Path(a).samefile(b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
