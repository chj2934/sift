"""Decide which bulk-ingested notes still earn their place in the vault.

Rationale (see project notes): the reasoning model already trained on the public
disclosed-report and NVD corpus, so retrieving it back adds little. What stays is
what the model *can't reliably reproduce* (detailed, bountied writeups) or *can't
know* (recent disclosures, actively-exploited CVEs) — plus everything the user
authored or curated.

Prune fails CLOSED. Only a `report` or `cve` note from a known bulk public corpus
(:data:`BULK_SOURCES`) can ever be dropped. Anything the user authored, any other type
(including one added later), any unrecognised source and any undated note is kept.
The old rules failed open: `prune --yes` deleted every `tool` note, the user's own
pre-2025 HackerOne reports and undated notes saved through `remember`.

:func:`classify` is pure. :func:`apply_prune` does the I/O. It never unlinks: drops
move to a quarantine outside the vault, are tombstoned so bulk re-ingest can't bring
them back, and only their own chunks are deleted from the index.
:func:`restore_quarantine` undoes it.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sift.config import get_settings
from sift.quality import as_float, has_bounty, is_user_authored, score_note
from sift.vault.notes import Note

log = logging.getLogger(__name__)

# CVE frontmatter tags that mark real-world exploitation.
_KEV_TAGS = frozenset({"kev", "known-exploited"})
# EPSS percentile at/above which we keep an otherwise-old CVE.
_EPSS_KEEP = 0.88
# A recently-catalogued CVE is only worth keeping if it also carries an
# exploitability signal — otherwise the reasoning model already covers it.
_EPSS_RECENT = 0.60
_RECENT_SEVERITIES = frozenset({"high", "critical"})
# Always keep — hand-authored, freshness-sourced, or primary vendor material. A
# `reference` note is the vendor's own wording pinned to a revision: the model can
# paraphrase a severity guideline but cannot quote this checkout's copy of it. A
# `tool` note is the user's per-program setup, the cheapest note in the vault.
_KEEP_TYPES = frozenset({"technique", "target", "finding", "writeup", "reference", "tool"})
# The only types prune judges at all. Any other type is kept, including one added
# to the schema later: an unknown type used to fall through to "drop".
_PRUNABLE_TYPES = frozenset({"report", "cve"})
# The only sources prune may drop from: public corpora the model trained on, which
# can be re-fetched at will. An allowlist, so a source prune doesn't recognise (a
# local note's custom `source`, or a remembered note whose caller set `source`) is kept.
BULK_SOURCES = frozenset({"nvd", "hackerone-public", "hackerone-hacktivity", "cisa-kev"})

UNDATED = "kept: undated"

# Windows: an AV scanner, the search indexer or an editor can hold a file for a moment.
_MOVE_ATTEMPTS = 5


@dataclass
class Verdict:
    keep: bool
    reason: str  # short bucket, for the summary table


def _year(note: Note) -> int | None:
    d = note.meta.created
    return d.year if d else None


def classify(note: Note, *, keep_since_year: int, report_quality_bar: int) -> Verdict:
    meta = note.meta
    t = meta.type
    if t in _KEEP_TYPES:
        return Verdict(True, "authored/curated")
    if t not in _PRUNABLE_TYPES:
        return Verdict(True, f"kept: {t} not prunable")
    if is_user_authored(meta):
        return Verdict(True, "user-authored")
    source = meta.source or ""
    if source not in BULK_SOURCES:
        return Verdict(True, f"kept: non-bulk source ({source or 'none'})")

    yr = _year(note)
    extra = meta.extra or {}

    if t == "report":
        if yr is None:
            # Unknown, not old: an ingester that failed to parse a date. Checked before
            # the bounty rule so the bucket says so instead of calling it "old".
            return Verdict(True, UNDATED)
        if yr >= keep_since_year:
            return Verdict(True, f"report {keep_since_year}+")
        if has_bounty(meta) and score_note(meta, note.body) >= report_quality_bar:
            return Verdict(True, "old but bountied+substantial")
        return Verdict(False, "old thin/dupe report")

    # t == "cve"
    if _KEV_TAGS.intersection(meta.tags):
        return Verdict(True, "KEV / known-exploited")
    pct = as_float(extra.get("epss_percentile"))
    if pct >= _EPSS_KEEP:
        return Verdict(True, f"EPSS pct >= {_EPSS_KEEP}")
    if yr is None:
        return Verdict(True, UNDATED)
    if yr >= keep_since_year:
        if (meta.severity or "") in _RECENT_SEVERITIES or pct >= _EPSS_RECENT:
            return Verdict(True, f"CVE {keep_since_year}+ w/ severity/EPSS")
        return Verdict(False, f"CVE {keep_since_year}+ but low-signal")
    return Verdict(False, "old low-signal CVE")


def refuse_reason(note: Note) -> str | None:
    """Why `apply_prune` must not touch this note, or None if it may.

    The hard invariants behind :func:`classify`, re-checked at the I/O boundary, so a
    caller that hands over the wrong list still can't move the user's notes.
    """
    meta = note.meta
    if meta.type not in _PRUNABLE_TYPES:
        return f"type {meta.type!r} is never pruned"
    if is_user_authored(meta):
        return "user-authored"
    if (meta.source or "") not in BULK_SOURCES:
        return f"source {meta.source!r} is not a bulk corpus"
    if meta.created is None:
        return "undated"
    return None


# --------------------------------------------------------------------------- #
# I/O: quarantine, tombstones, index
# --------------------------------------------------------------------------- #
@dataclass
class PruneResult:
    quarantine: Path  # where moved notes went (created only if something moved)
    moved: list[str] = field(default_factory=list)  # ids now in quarantine
    already_gone: list[str] = field(default_factory=list)  # file vanished before the move
    refused: list[str] = field(default_factory=list)  # "id: why", left untouched
    failed: list[str] = field(default_factory=list)  # "path: error", still in the vault
    tombstoned: int = 0  # new ledger entries
    unindexed: list[str] = field(default_factory=list)  # ids whose chunks were deleted
    # Stored paths deleted one file at a time: a moved twin whose id stays in the vault.
    unindexed_paths: list[str] = field(default_factory=list)
    rows_deleted: int | None = None  # index rows removed, when the Store reports it
    index_error: str | None = None  # the files are safe either way; see apply_prune
    tombstone_error: str | None = None


@dataclass
class RestoreResult:
    restored: list[str] = field(default_factory=list)  # vault-relative paths moved back
    conflicts: list[str] = field(default_factory=list)  # vault already has that path
    failed: list[str] = field(default_factory=list)
    untombstoned: int = 0


def quarantine_dir(stamp: str | None = None) -> Path:
    """`<parent of SIFT_DB_PATH>/pruned/<stamp>/`, i.e. `data/pruned/...` by default.

    Outside the vault, because `iter_notes` rglobs every subfolder and would index a
    quarantine kept inside it. In the default layout `.gitignore` already covers it
    (`/data/`), so thousands of private notes can't be committed by a `git add -A`;
    a sibling `vault.pruned/` would not have been.
    """
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    return get_settings().resolved_db().parent / "pruned" / stamp


def _claim_dir(base: Path) -> Path:
    """Create a quarantine dir no earlier run used (two prunes in one second)."""
    base.parent.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        cand = base if n == 1 else base.with_name(f"{base.name}-{n}")
        try:
            cand.mkdir()
            return cand
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a fresh quarantine dir next to {base}")


def _free_target(p: Path) -> Path:
    if not p.exists():
        return p
    for n in range(2, 1000):
        cand = p.with_name(f"{p.stem} ({n}){p.suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"no free name for {p}")


def _move(src: Path, dst: Path) -> None:
    """Move one file without ever leaving it in both places or in neither.

    A same-volume rename is atomic. Only for a cross-volume move does it copy, and then
    the copy is removed again if the source can't be deleted. Plain `shutil.move` would
    leave the copy behind after a locked-source failure on Windows.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(_MOVE_ATTEMPTS):
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            if attempt == _MOVE_ATTEMPTS - 1:
                raise
            time.sleep(0.1 * (attempt + 1))
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(src, dst)
            try:
                os.unlink(src)
            except OSError:
                dst.unlink(missing_ok=True)
                raise
            return


def _open_store():
    from sift.index.store import Store  # lancedb is heavy; only load it when deleting

    return Store()


def _vault_write_lock(vault: Path) -> contextlib.AbstractContextManager:
    """The vault's write lock (this process and the MCP server), so a concurrent save
    can't rewrite a file mid-move."""
    from sift.vault import notes

    return notes.write_lock(vault)


def _norm_path(p: str | os.PathLike[str]) -> str:
    # Same canonical spelling as index.store.norm_path: stored paths are compared, not
    # rewritten, so separators and drive-letter case can't hide a match on Windows.
    return os.path.normcase(os.path.abspath(os.fspath(p)))


def _unindex(store, ids: list[str], paths: list[str]) -> tuple[int | None, list[str]]:
    """Delete whole notes by id, and single files by stored path.

    Returns (rows deleted when the Store reports it, stored paths deleted). A stand-in
    store without `delete_paths`/`path_state` deletes no paths; the reindex sweep reaps
    those.
    """
    rows: int | None = None
    if ids:
        n = store.delete_notes(ids)  # one commit for the whole batch
        rows = n if isinstance(n, int) else None
    done: list[str] = []
    by_path = getattr(store, "delete_paths", None)
    path_state = getattr(store, "path_state", None)
    if paths and callable(by_path) and callable(path_state):
        want = {_norm_path(p) for p in paths}
        done = sorted(sp for sp in path_state() if sp and _norm_path(sp) in want)
        if done:
            n = by_path(done)
            if isinstance(n, int):
                rows = (rows or 0) + n
    return rows, done


def apply_prune(
    drop: Iterable[Note],
    *,
    vault: Path | None = None,
    keep_ids: Iterable[str] = (),
    dest: Path | None = None,
    store=None,
    reason: str = "sift prune",
) -> PruneResult:
    """Quarantine `drop`, tombstone it, and delete exactly its chunks from the index.

    - Each note is re-checked with :func:`refuse_reason` before it's touched. One that
      fails the check is left alone and listed in `refused`.
    - Files are moved, never unlinked, into `dest` (default :func:`quarantine_dir`),
      keeping their vault-relative path so two notes sharing a basename can't collide.
      A move that fails (Windows lock, AV scan) leaves that note in the vault, indexed
      and untombstoned. The loop carries on with the rest.
    - Tombstones (id + url + source) are written for every note that left the vault,
      even when the loop is interrupted part-way.
    - The vault's write lock is held across the moves and the ledger write.
    - Chunks are deleted for ids that left the vault (or whose file had already gone),
      in one batched delete, so no forced re-embed is needed afterwards. An id that
      `keep_ids` or a note left behind still carries is never deleted by id. KEV and
      NVD twins share a CVE id, so that would empty the kept twin's chunks. Only the
      departed file's own rows go, by stored path, when the Store supports it.
    - An index failure goes into `index_error` instead of being raised: the files
      are already safe in quarantine. Run an incremental reindex afterwards.
    """
    notes = list(drop)
    vault = (vault or get_settings().resolved_vault()).resolve()
    auto_dest = dest is None
    target_root = (dest or quarantine_dir()).resolve()
    res = PruneResult(quarantine=target_root)
    if not notes:
        return res
    if target_root.is_relative_to(vault):
        raise ValueError(
            f"quarantine {target_root} is inside the vault {vault}; iter_notes would index it again"
        )

    left: list[Note] = []  # moved out of the vault by this call
    gone: list[Note] = []  # file had already vanished; its rows are orphans
    still_present: set[str] = set()  # ids of drop candidates that stayed put
    claimed = False
    # Held across the moves and the ledger write, so a save (MCP `remember`, an ingest)
    # can't rewrite a file mid-move or slip in between the move and its tombstone.
    with _vault_write_lock(vault):
        try:
            for note in notes:
                nid = note.meta.id
                why = refuse_reason(note)
                if why is None and note.path is None:
                    why = "no path on disk"
                if why is not None:
                    res.refused.append(f"{nid}: {why}")
                    still_present.add(nid)
                    continue
                src = Path(note.path)
                if not src.exists():
                    res.already_gone.append(nid)
                    gone.append(note)
                    continue
                try:
                    rel = src.resolve().relative_to(vault)
                except ValueError:
                    res.refused.append(f"{nid}: {src} is outside the vault")
                    still_present.add(nid)
                    continue
                if not claimed:
                    if auto_dest:
                        target_root = _claim_dir(target_root)
                    else:
                        target_root.mkdir(parents=True, exist_ok=True)
                    res.quarantine = target_root
                    claimed = True
                try:
                    _move(src, _free_target(target_root / rel))
                except OSError as exc:
                    res.failed.append(f"{src}: {exc}")
                    still_present.add(nid)
                    log.warning("prune: could not move %s: %s", src, exc)
                    continue
                res.moved.append(nid)
                left.append(note)
        finally:
            if left:
                try:
                    from sift.tombstones import record_note_tombstones

                    res.tombstoned = record_note_tombstones(left, reason=reason)
                except Exception as exc:  # noqa: BLE001 - surfaced on the result
                    res.tombstone_error = f"{type(exc).__name__}: {exc}"
                    log.error("prune: tombstones not recorded: %s", res.tombstone_error)

    protected = set(keep_ids) | still_present
    ids = sorted({n.meta.id for n in [*left, *gone]} - protected)
    paths = sorted({str(n.path) for n in [*left, *gone] if n.meta.id in protected})
    if ids or paths:
        try:
            st = store if store is not None else _open_store()
            res.rows_deleted, res.unindexed_paths = _unindex(st, ids, paths)
            res.unindexed = ids
        except Exception as exc:  # noqa: BLE001 - surfaced on the result
            res.index_error = f"{type(exc).__name__}: {exc}"
            log.error("prune: index cleanup failed: %s", res.index_error)
    log.info(
        "prune: moved %d to %s (%d refused, %d failed, %d already gone), "
        "%d new tombstones, %d ids + %d twin paths unindexed",
        len(res.moved),
        res.quarantine,
        len(res.refused),
        len(res.failed),
        len(res.already_gone),
        res.tombstoned,
        len(res.unindexed),
        len(res.unindexed_paths),
    )
    return res


def restore_quarantine(src: Path, *, vault: Path | None = None) -> RestoreResult:
    """Undo :func:`apply_prune`: move each note under `src` back and clear its tombstones.

    Never overwrites. A vault path that is in use again stays in quarantine and is
    listed in `conflicts`. Run an incremental reindex afterwards to index the notes again.
    """
    from sift.tombstones import remove_tombstones
    from sift.vault.notes import load_note

    vault = vault or get_settings().resolved_vault()
    src = Path(src)
    v, s = vault.resolve(), src.resolve()
    if s.is_relative_to(v) or v.is_relative_to(s):
        # Pointed at the vault by mistake: the cleanup below would rmdir its empty folders.
        raise ValueError(f"{src} overlaps the vault {vault}; pass a quarantine folder")
    res = RestoreResult()
    ids: list[str] = []
    urls: list[str] = []
    # Locked so a save can't claim a vault path between the exists() check and the move.
    with _vault_write_lock(v):
        for path in sorted(src.rglob("*.md")):
            rel = path.relative_to(src)
            target = vault / rel
            if target.exists():
                res.conflicts.append(str(rel))
                continue
            try:
                meta = load_note(path).meta
            except Exception:  # noqa: BLE001 - restore the file anyway, just without a tombstone key
                meta = None
            try:
                _move(path, target)
            except OSError as exc:
                res.failed.append(f"{rel}: {exc}")
                continue
            res.restored.append(str(rel))
            if meta is not None:
                ids.append(meta.id)
                if meta.url:
                    urls.append(meta.url)
        if ids or urls:
            res.untombstoned = remove_tombstones(ids=ids, urls=urls)
    # Drop the emptied folders; anything left (conflicts, failures) keeps its dir.
    for d in [*sorted((p for p in src.rglob("*") if p.is_dir()), reverse=True), src]:
        with contextlib.suppress(OSError):  # not empty: something stayed in quarantine
            d.rmdir()
    return res
