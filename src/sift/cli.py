"""sift command-line interface."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

# Windows terminals default to cp1252 and mangle em-dashes / box chars in note text.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from sift.config import get_settings
from sift.vault.schema import NOTE_TYPES

# No locals in tracebacks: a frame's locals can hold settings with API tokens.
app = typer.Typer(
    no_args_is_help=True, add_completion=False, help=__doc__, pretty_exceptions_show_locals=False
)
ingest_app = typer.Typer(no_args_is_help=True, help="Pull external data into the vault.")
app.add_typer(ingest_app, name="ingest")
distill_app = typer.Typer(no_args_is_help=True, help="Gate source material into technique notes.")
app.add_typer(distill_app, name="distill")
trash_app = typer.Typer(
    no_args_is_help=True,
    help="Notes soft-deleted into the vault's .trash folder (MCP forget_note).",
)
app.add_typer(trash_app, name="trash")
console = Console()
err = Console(stderr=True)  # warnings and errors: stdout stays the command's output


# --------------------------------------------------------------------------- #
# logging: library modules log, never print (stdout is the MCP server's wire)
# --------------------------------------------------------------------------- #
_OWN_HANDLER = "_sift_cli_handler"


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever `sys.stderr` is when a record is emitted - a progress bar's
    proxy, a test runner's capture - and never to stdout."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property  # type: ignore[override]
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, _value) -> None:  # StreamHandler.__init__/setStream assign it
        pass


class _CliFormatter(logging.Formatter):
    """Progress lines as they are; warnings and errors labelled."""

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{record.levelname.lower()}: {text}"
        return text


def _configure_logging(*, serving: bool) -> None:
    """Send the ``sift`` loggers to stderr. INFO for CLI commands (ingest and reindex
    progress); WARNING while serving MCP, where stderr is the client's error log.

    Only the ``sift`` logger is configured, never the root: a root INFO level would
    surface httpx's per-request lines during every ingest.
    """
    lg = logging.getLogger("sift")
    for h in list(lg.handlers):
        if getattr(h, _OWN_HANDLER, False):
            lg.removeHandler(h)
    handler = _StderrHandler()
    setattr(handler, _OWN_HANDLER, True)
    if serving:
        handler.setFormatter(logging.Formatter("sift %(levelname)s %(name)s: %(message)s"))
        lg.setLevel(logging.WARNING)
    else:
        handler.setFormatter(_CliFormatter("%(message)s"))
        lg.setLevel(logging.INFO)
    lg.addHandler(handler)
    lg.propagate = False


@app.callback()
def _main(ctx: typer.Context) -> None:
    _configure_logging(serving=ctx.invoked_subcommand == "mcp")


@contextlib.contextmanager
def _bar(description: str, *, total: int | None = None):
    """Progress display. A real percentage when the total is known (reindex), a
    running count when it isn't (ingest sources are network generators).

    Yields (progress, task_id). Falls back to a plain line when stdout is not a
    terminal, so piped output and CI logs stay readable.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    if not console.is_terminal:
        console.print(f"{description}…")

        class _Null:
            def update(self, *a, **k):
                pass

        yield _Null(), None
        return

    columns = [SpinnerColumn(), TextColumn("[progress.description]{task.description}")]
    if total:
        columns += [
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ]
    else:
        columns += [MofNCompleteColumn(), TimeElapsedColumn()]

    with Progress(*columns, console=console, transient=False) as progress:
        yield progress, progress.add_task(description, total=total)


def _titled(progress, task, name: str):
    """An ingest `on_progress` callback: count plus the current title. Titles are
    escaped - the bar renders its description as markup, and a title like '[/x]'
    used to raise mid-ingest."""
    return lambda n, title: progress.update(
        task, completed=n, description=f"ingest {name}: {escape(str(title)[:44])}"
    )


def _fail(message: str) -> typer.Exit:
    err.print(f"[red]{escape(message)}[/]")
    return typer.Exit(1)


# --------------------------------------------------------------------------- #
# init / status / doctor
# --------------------------------------------------------------------------- #
@app.command()
def init() -> None:
    """Create the vault folder structure and a starter config."""
    s = get_settings()
    vault = s.resolved_vault()
    for t in NOTE_TYPES:
        (vault / t).mkdir(parents=True, exist_ok=True)
    (vault / "inbox").mkdir(parents=True, exist_ok=True)
    (vault / ".gitkeep").touch()
    s.resolved_db().mkdir(parents=True, exist_ok=True)

    env = s.model_config.get("env_file")
    if env and not Path(env).exists():
        example = Path(env).with_name(".env.example")
        if example.exists():
            Path(env).write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
            console.print(f"[green]wrote[/] {env} (from .env.example — fill in your keys)")

    console.print(f"[green]vault ready[/] at {vault}")
    console.print('next: [bold]sift ingest kev[/] then [bold]sift search "..."[/]')


def _print_index_status(s) -> None:
    """Index size, fragment backlog, keyword-index coverage and build record."""
    try:
        from sift.index.store import Store

        store = Store()
        info = store.index_info()
    except Exception as exc:  # noqa: BLE001 - e.g. an unknown model's dimension
        err.print(f"[red]index unavailable:[/] {escape(str(exc))}")
        return
    console.print(f"index chunks: [bold]{info.get('rows') or 0}[/]  ({s.resolved_db()})")
    if not info.get("exists"):
        console.print("  [dim]no index yet - run `sift reindex`[/]")
        return
    frags = info.get("fragments") or 0
    unindexed = info.get("fts_unindexed_rows")
    console.print(
        f"  {frags} fragments, version {info.get('version')}, keyword index "
        f"{'yes' if info.get('fts') else '[yellow]missing[/]'}"
        + (f" ({unindexed} rows not folded in yet)" if unindexed else "")
    )
    if frags > Store.COMPACT_MAX_ROUTINE_FRAGMENTS:
        console.print(
            f"  [yellow]{frags} fragments slow every search:[/] run [bold]sift compact[/] once"
        )
    try:
        from sift.pipeline import index_meta, stale_index_reason

        meta = index_meta(store)
        if meta:
            console.print(
                f"  built by chunker {meta.get('chunker_version')} with {meta.get('embed_model')}"
            )
        reason = stale_index_reason(store)
        if reason:
            console.print(f"  [yellow]{escape(reason)}:[/] run [bold]sift reindex --force[/] once")
    except Exception as exc:  # noqa: BLE001 - the counts above are still right
        err.print(f"[yellow]could not read the index build record:[/] {escape(str(exc))}")


@app.command()
def status() -> None:
    """Show note counts, index health, and the last ingest runs."""
    from sift.ingest.base import load_state

    s = get_settings()
    vault = s.resolved_vault()

    # Counted from the vault catalog (a stat walk; only new or changed files are
    # parsed), not by parsing all ~14k notes on every call.
    cat = None
    counts: dict[str, int] = {}
    if vault.is_dir():
        from sift.vault.catalog import fresh_catalog

        cat = fresh_catalog(vault)
        counts = cat.counts_by_type()

    t = Table(title="vault")
    t.add_column("type")
    t.add_column("notes", justify="right")
    for k in NOTE_TYPES:
        t.add_row(k, str(counts.get(k, 0)))
    for k in sorted(set(counts) - set(NOTE_TYPES)):
        t.add_row(f"[dim]{escape(k)}", str(counts[k]))
    t.add_row("[bold]total", f"[bold]{sum(counts.values())}")
    console.print(t)
    if cat is None:
        console.print(f"[yellow]no vault at {vault}[/] - run `sift init`")
    else:
        skipped, dups = cat.skipped(), cat.duplicate_ids()
        if skipped:
            console.print(
                f"[yellow]{len(skipped)} note file(s) can't be read[/] (empty, no frontmatter "
                "or invalid): [bold]sift doctor[/] lists them"
            )
        if dups:
            console.print(
                f"[yellow]{len(dups)} id(s) are carried by more than one file:[/] "
                "[bold]sift doctor[/] lists them"
            )

    _print_index_status(s)
    query = f" (queries on {s.query_device})" if s.query_device else ""
    rerank = f"on ({s.rerank_model})" if s.rerank else "off"
    console.print(f"embeddings: {s.embed_model} on {s.embed_device}{query}   rerank: {rerank}")

    state = load_state()
    if state:
        st = Table(title="last ingest")
        for col, justify in (
            ("source", "left"),
            ("when", "left"),
            ("new", "right"),
            ("updated", "right"),
            ("unchanged", "right"),
            ("errors", "right"),
            ("run", "left"),
        ):
            st.add_column(col, justify=justify)  # type: ignore[arg-type]
        for src, info in sorted(state.items()):
            if not isinstance(info, dict):
                continue
            run = "ok"
            if info.get("complete") is False:
                run = f"[yellow]incomplete[/] ({escape(str(info.get('aborted') or 'aborted'))})"
                if info.get("last_complete"):
                    run += f", last complete {escape(str(info['last_complete']))}"
            st.add_row(
                escape(src),
                escape(str(info.get("last_run", "?"))),
                str(info.get("written", "?")),
                str(info.get("updated", "-")),
                str(info.get("unchanged", "-")),
                str(info.get("errors", "-")),
                run,
            )
        console.print(st)


@app.command()
def doctor(
    index: bool = typer.Option(
        True, "--index/--no-index", help="Also compare the index with the vault files."
    ),
) -> None:
    """Report vault and index problems. Read-only: changes nothing.

    Lists ids carried by more than one file (each file's source and url, and whether
    the bodies are identical), note files that can't be read, and index rows whose
    file is gone. Merge or delete duplicates by hand after reading the bodies; nothing
    here deletes or merges a note.
    """
    import hashlib

    from sift.vault.catalog import fresh_catalog
    from sift.vault.notes import load_note

    vault = get_settings().resolved_vault()
    if not vault.is_dir():
        raise _fail(f"no vault at {vault}")
    cat = fresh_catalog(vault)

    def rel(p: Path) -> str:
        try:
            return p.relative_to(vault).as_posix()
        except ValueError:
            return str(p)

    dups = cat.duplicate_ids()
    if dups:
        t = Table(title=f"{len(dups)} id(s) carried by more than one file")
        for col in ("id", "file", "source", "url", "bodies"):
            t.add_column(col, overflow="fold")
        for nid, _paths in dups.items():
            rows = cat.by_id(nid)
            digests = set()
            for row in rows:
                try:
                    digests.add(hashlib.sha1(load_note(row.path).body.strip().encode()).hexdigest())
                except Exception:  # noqa: BLE001
                    digests.add(f"unreadable:{row.path}")
            same = "identical" if len(digests) == 1 else "differ"
            for i, row in enumerate(rows):
                t.add_row(
                    escape(nid) if i == 0 else "",
                    escape(rel(row.path)),
                    escape(row.source or ""),
                    escape(row.url or ""),
                    same if i == 0 else "",
                )
        console.print(t)
        console.print(
            "[dim]Both files of a pair are indexed. Judge the bodies, then merge or delete "
            "by hand - never on the title alone.[/]"
        )
    else:
        console.print("[green]no duplicate ids[/]")

    skipped = cat.skipped()
    if skipped:
        t = Table(title=f"{len(skipped)} note file(s) that can't be read (skipped everywhere)")
        t.add_column("file", overflow="fold")
        t.add_column("why", overflow="fold")
        for path, why in skipped:
            t.add_row(escape(rel(path)), escape(why))
        console.print(t)
    else:
        console.print("[green]every note file reads[/]")

    if not index:
        return
    try:
        from sift.index.store import Store, norm_path

        files = Store().indexed_files()
    except Exception as exc:  # noqa: BLE001
        err.print(f"[yellow]index not checked:[/] {escape(str(exc))}")
        return
    on_disk = {norm_path(r.path) for r in cat.rows()}
    indexed = {norm_path(p) for _nid, p, _m in files if p}
    gone = sorted({(nid, p) for nid, p, _m in files if p and not os.path.exists(p)})
    no_path = sorted({nid for nid, p, _m in files if not p})
    not_indexed = len(on_disk - indexed)
    if gone:
        console.print(
            f"[yellow]{len(gone)} indexed file(s) are no longer on disk[/] "
            "(search still returns them): run [bold]sift reindex[/]"
        )
        for nid, p in gone[:10]:
            console.print(f"  {escape(nid)}  [dim]{escape(p)}[/]")
        if len(gone) > 10:
            console.print(f"  [dim]... and {len(gone) - 10} more[/]")
    if no_path:
        console.print(
            f"[yellow]{len(no_path)} id(s) have index rows without a file path[/] "
            "(an old index): one [bold]sift reindex --force[/] rewrites them"
        )
    if not_indexed:
        console.print(
            f"[yellow]{not_indexed} note file(s) are not indexed yet:[/] run [bold]sift reindex[/]"
        )
    if not (gone or no_path or not_indexed):
        console.print("[green]index matches the vault[/]")


# --------------------------------------------------------------------------- #
# reindex / compact / prune / search
# --------------------------------------------------------------------------- #
def _check_embed_width() -> None:
    """Refuse a forced rebuild the configured dimension can't hold.

    `--force` drops the table before embedding anything, so a model whose vectors
    don't fit SIFT_EMBED_DIM (or the dimension derived from the model) used to leave
    an empty index behind.
    """
    from sift.index import embed

    s = get_settings()
    try:
        dim = s.effective_embed_dim()
    except ValueError as exc:
        raise _fail(str(exc)) from None
    try:
        width = len(embed.get_embedder().embed_one("dimension probe"))
    except Exception as exc:  # noqa: BLE001
        raise _fail(f"could not load the embedding model {s.embed_model}: {exc}") from None
    if width != dim:
        raise _fail(
            f"{s.embed_model} produces {width}-dim vectors but the index would be built for "
            f"{dim} (SIFT_EMBED_DIM, or the model's known size). Fix SIFT_EMBED_DIM; nothing "
            "was dropped."
        )


def _print_reindex_stats(stats, *, label: str = "done") -> None:
    unchanged = f"  [dim]({stats.unchanged} unchanged, skipped)[/]" if stats.unchanged else ""
    by_type = ", ".join(f"{k}:{v}" for k, v in sorted(stats.by_type.items()))
    console.print(
        f"[green]{label}[/] {stats.notes} notes, {stats.chunks} chunks"
        + (f" ({by_type})" if by_type else "")
        + unchanged
    )
    if stats.removed:
        console.print(f"  removed the index rows of {stats.removed} deleted or emptied note(s)")
    if stats.skipped:
        console.print(f"  [dim]{stats.skipped} note(s) have no indexable text[/]")
    if stats.unreadable:
        err.print(
            f"[yellow]{stats.unreadable} note file(s) failed to load; their index rows were "
            "kept.[/] [bold]sift doctor[/] lists them."
        )
    if stats.reap_refused:
        err.print(
            f"[yellow]kept the index rows of {stats.reap_refused} missing note(s):[/] this looks "
            "like a mass delete. Is the vault drive mounted and SIFT_VAULT_PATH right? If they "
            "really were deleted, re-run with [bold]--allow-mass-reap[/]."
        )
    if stats.requeued:
        console.print(
            f"  [dim]{stats.requeued} note(s) changed while being embedded; the next reindex "
            "picks them up[/]"
        )
    if stats.duplicate_ids:
        err.print(
            f"[yellow]{stats.duplicate_ids} id(s) are carried by more than one file[/] (both "
            "are indexed): [bold]sift doctor[/] lists them."
        )
    if stats.stale_index:
        err.print(
            f"[yellow]{escape(stats.stale_index)}:[/] run [bold]sift reindex --force[/] once."
        )
    maintenance = stats.maintenance or {}
    if maintenance.get("error"):
        err.print(
            f"[red]index maintenance failed:[/] {escape(str(maintenance['error']))} "
            "(the data is committed and searchable)"
        )
    elif maintenance.get("ran"):
        console.print(
            f"  [dim]compacted {maintenance.get('before', {}).get('fragments')} -> "
            f"{maintenance.get('after', {}).get('fragments')} fragments[/]"
        )


def _rescore() -> None:
    """Recompute every note's quality score and update the stored ones in place."""
    from sift.index.store import Store
    from sift.quality import score_note
    from sift.vault.notes import iter_notes

    vault = get_settings().resolved_vault()
    scores: dict[str, int] = {}
    for note in iter_notes(vault):
        # Twins (a KEV and an NVD file for one CVE) share an id and one stored score.
        q = score_note(note.meta, note.body)
        scores[note.meta.id] = max(scores.get(note.meta.id, 0), q)
    try:
        store = Store()
        changed = store.rescore(scores)
    except Exception as exc:  # noqa: BLE001
        raise _fail(f"rescore failed: {exc}") from None
    console.print(
        f"[green]rescored[/] {changed} of {len(scores)} notes (quality only, nothing re-embedded)"
    )
    if changed:
        report = store.optimize()
        if report.get("error"):
            err.print(f"[red]index maintenance failed:[/] {escape(str(report['error']))}")


@app.command()
def reindex(
    force: bool = typer.Option(
        False, "--force", help="Drop and rebuild the whole index (after a model or chunker change)."
    ),
    allow_mass_reap: bool = typer.Option(
        False,
        "--allow-mass-reap",
        help="Remove the rows of missing files even when it looks like a mass delete "
        "(only after checking the vault is mounted and SIFT_VAULT_PATH is right).",
    ),
    rescore: bool = typer.Option(
        False,
        "--rescore",
        help="Only recompute stored quality scores (no re-embedding), after a scoring change.",
    ),
) -> None:
    """Bring the index in line with the markdown vault.

    Incremental by default: only new and changed notes are embedded, and the rows of
    deleted or emptied notes are removed.
    """
    if rescore:
        if force:
            raise _fail("--rescore and --force don't combine: --force re-embeds everything")
        _rescore()
        return

    from sift.pipeline import count_notes
    from sift.pipeline import reindex as _reindex

    if force:
        _check_embed_width()
    total = count_notes()
    try:
        with _bar(f"reindexing {total} notes", total=total) as (progress, task):
            stats = _reindex(
                force=force,
                allow_mass_reap=allow_mass_reap,
                on_progress=lambda done: progress.update(task, completed=done),
            )
            # The walk is the truth: notes added or removed since count_notes().
            progress.update(task, total=stats.walked, completed=stats.walked)
    except (RuntimeError, ValueError) as exc:  # unreadable index, model/dimension mismatch
        raise _fail(str(exc)) from None
    _print_reindex_stats(stats)


def _human_bytes(n: object) -> str:
    if not isinstance(n, int | float):
        return "-"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@app.command()
def compact(
    retain_minutes: int | None = typer.Option(
        None,
        "--retain-minutes",
        help="Keep index versions younger than this many minutes (default 60).",
    ),
    unsafe_zero_retention: bool = typer.Option(
        False,
        "--unsafe-zero-retention",
        help="Allow a retention under 10 minutes (default 0 with this flag). Stop every sift "
        "process first, Claude Code's MCP servers included: a reader of a pruned version fails.",
    ),
    rebuild_fts: bool = typer.Option(
        False, "--rebuild-fts", help="Also rebuild the keyword (FTS) index from scratch."
    ),
) -> None:
    """Compact the index and reclaim the disk used by old versions.

    Ingests and reindex compact routinely, but skip a long-neglected table: the first
    run on one takes minutes and rewrites the table once, so check free disk space.
    With the default retention it is safe while an MCP server is running; don't run an
    ingest or reindex at the same time.
    """
    from datetime import timedelta

    from sift.index.store import Store

    if unsafe_zero_retention:
        minutes = retain_minutes if retain_minutes is not None else 0
        err.print(
            "[yellow]unsafe retention:[/] every sift process (MCP servers included) must be "
            "stopped, or its next read may fail."
        )
    else:
        minutes = 60 if retain_minutes is None else retain_minutes
        if minutes < 10:
            raise _fail(
                f"--retain-minutes {minutes} is under 10: a running MCP server may still read "
                "those versions. Pass --unsafe-zero-retention with every sift process stopped."
            )
    if minutes < 0:
        raise _fail("--retain-minutes must not be negative")
    try:
        store = Store()
        with (
            console.status("compacting the index...")
            if console.is_terminal
            else contextlib.nullcontext()
        ):
            report = store.optimize(
                timedelta(minutes=minutes),
                force=True,
                rebuild_fts=rebuild_fts,
                measure_disk=True,
                allow_unsafe_retain=unsafe_zero_retention,
            )
    except ValueError as exc:
        raise _fail(str(exc)) from None
    if not report.get("before"):
        if report.get("error"):
            raise _fail(f"compaction failed: {report['error']}")
        console.print(f"nothing to compact: {report.get('reason') or 'no index yet'}")
        return
    before, after = report["before"], report.get("after") or {}
    t = Table(title=f"compact (retention {minutes} min)")
    t.add_column("")
    t.add_column("before", justify="right")
    t.add_column("after", justify="right")
    for key, label in (("fragments", "fragments"), ("version", "version"), ("rows", "rows")):
        t.add_row(label, str(before.get(key, "-")), str(after.get(key, "-")))
    t.add_row("disk", _human_bytes(before.get("disk_bytes")), _human_bytes(after.get("disk_bytes")))
    console.print(t)
    console.print(f"took {report.get('seconds', 0)} s")
    if report.get("error"):
        raise _fail(f"compaction failed: {report['error']} (the data is committed)")


def _restore_prune(folder: str) -> None:
    from sift.prune import restore_quarantine

    src = Path(folder)
    if not src.is_dir():
        raise _fail(f"no such quarantine folder: {folder}")
    try:
        res = restore_quarantine(src)
    except ValueError as exc:
        raise _fail(str(exc)) from None
    console.print(
        f"[green]restored {len(res.restored)} note(s)[/], {res.untombstoned} tombstone(s) cleared"
    )
    if res.conflicts:
        err.print(
            f"[yellow]{len(res.conflicts)} left in quarantine:[/] the vault already has a file there"
        )
        for c in res.conflicts[:10]:
            err.print(f"  {escape(c)}")
    if res.failed:
        err.print(f"[red]{len(res.failed)} could not be moved back:[/]")
        for f in res.failed[:10]:
            err.print(f"  {escape(f)}")
    if res.restored:
        console.print("next: [bold]sift reindex[/] to index them again")
    if res.failed:
        raise typer.Exit(1)


@app.command()
def prune(
    keep_since: int = typer.Option(2025, help="Keep reports/CVEs created in this year or later."),
    report_quality_bar: int = typer.Option(
        62, help="Older reports are kept only if bountied AND scoring at least this."
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Move the dropped notes to data/pruned/<stamp> and remove them from the index "
        "(default: dry run).",
    ),
    restore: str | None = typer.Option(
        None,
        "--restore",
        help="Undo a prune: move the notes in this quarantine folder back into the vault.",
    ),
) -> None:
    """Drop bulk-ingested notes the reasoning model already knows; keep the rest.

    Only dated report/cve notes from bulk sources (nvd, hackerone-public,
    hackerone-hacktivity, cisa-kev) can go; anything you wrote is kept. Default is a
    dry run. --yes moves the drops to a quarantine folder outside the vault,
    tombstones them so a re-ingest doesn't bring them back, and deletes their index
    rows (no re-embed). --restore FOLDER undoes it.
    """
    if restore is not None:
        _restore_prune(restore)
        return

    from collections import Counter

    from sift.prune import classify
    from sift.vault.notes import iter_notes

    vault = get_settings().resolved_vault()
    keep: list = []
    drop: list = []
    keep_reasons: Counter = Counter()
    drop_reasons: Counter = Counter()

    for note in iter_notes(vault):
        v = classify(note, keep_since_year=keep_since, report_quality_bar=report_quality_bar)
        (keep if v.keep else drop).append(note)
        (keep_reasons if v.keep else drop_reasons)[v.reason] += 1

    t = Table(title=f"prune plan  (keep_since={keep_since})")
    t.add_column("action")
    t.add_column("reason")
    t.add_column("notes", justify="right")
    for r, n in keep_reasons.most_common():
        t.add_row("[green]keep", escape(r), str(n))
    for r, n in drop_reasons.most_common():
        t.add_row("[red]drop", escape(r), str(n))
    t.add_row("[bold]total", "", f"[bold]{len(keep) + len(drop)}")
    console.print(t)
    console.print(f"[green]keep {len(keep)}[/]   [red]drop {len(drop)}[/]")

    if drop:
        console.print("\n[dim]sample drops:[/]")
        for note in drop[:10]:
            console.print(f"  [dim]{note.meta.type}[/] {escape(note.meta.title[:80])}")

    if not yes:
        console.print(
            "\n[yellow]dry run[/] — re-run with [bold]--yes[/] to move the drops to quarantine "
            "(undo with [bold]sift prune --restore <folder>[/])."
        )
        return
    if not drop:
        console.print("nothing to drop")
        return

    from sift.prune import apply_prune

    try:
        # keep_ids: a dropped NVD twin must not delete the chunks of the KEV twin kept
        # under the same CVE id.
        res = apply_prune(drop, vault=vault, keep_ids={n.meta.id for n in keep})
    except (RuntimeError, ValueError) as exc:  # vault lock timeout, bad quarantine dir
        raise _fail(str(exc)) from None

    console.print(f"[red]moved {len(res.moved)} note(s)[/] to {res.quarantine}")
    if res.moved:
        console.print(f'  undo: [bold]sift prune --restore "{res.quarantine}"[/]')
    rows = "?" if res.rows_deleted is None else res.rows_deleted
    console.print(
        f"  {res.tombstoned} tombstoned (a re-ingest skips them), {rows} index rows removed"
    )
    if res.already_gone:
        console.print(f"  [dim]{len(res.already_gone)} were already gone from disk[/]")
    if res.refused:
        err.print(f"[yellow]{len(res.refused)} refused at the last check, left in place:[/]")
        for r in res.refused[:10]:
            err.print(f"  {escape(r)}")
    if res.failed:
        err.print(f"[yellow]{len(res.failed)} could not be moved (still in the vault, indexed):[/]")
        for f in res.failed[:10]:
            err.print(f"  {escape(f)}")
    if res.tombstone_error:
        err.print(f"[red]tombstones not recorded:[/] {escape(res.tombstone_error)}")
    if res.index_error:
        err.print(
            f"[red]index not updated:[/] {escape(res.index_error)}. The files are safe in "
            "quarantine; run [bold]sift reindex[/] to drop their rows."
        )
    if res.failed or res.tombstone_error or res.index_error:
        raise typer.Exit(1)


@app.command()
def search(
    query: str,
    k: int = typer.Option(8, "-k", help="Number of notes to return."),
    type: str | None = typer.Option(None, help="Filter by note type."),
    cwe: str | None = typer.Option(None, help="Filter by CWE, e.g. CWE-79."),
    program: str | None = typer.Option(None, help="Filter by program/vendor."),
    min_quality: int = typer.Option(
        0, "--min-quality", help="Drop hits below this 0-100 quality score."
    ),
    links: bool = typer.Option(False, "--links", help="Also show 1-hop linked notes."),
    full: bool = typer.Option(False, "--full", help="Print full excerpts."),
) -> None:
    """Hybrid search the memory."""
    from sift.pipeline import search as _search

    filters = {k2: v for k2, v in {"type": type, "cwe": cwe, "program": program}.items() if v}
    try:
        result = _search(
            query, k=k, filters=filters or None, expand_links=links, min_quality=min_quality
        )
    except (ValueError, RuntimeError) as exc:  # bad filter, model/index mismatch, search failure
        raise _fail(str(exc)) from None

    for w in result.warnings:
        err.print(f"[yellow]warning:[/] {escape(w)}")
    if not result.hits:
        console.print("[yellow]no matches[/] — is the index built? try `sift reindex`")
        raise typer.Exit(1)

    for i, h in enumerate(result.hits, 1):
        meta = "  ".join(
            x
            for x in [
                h.severity and f"sev:{h.severity}",
                h.program and f"@{h.program}",
                f"q:{h.quality}",
                f"score {h.score:.3f}",
                f"id {h.note_id}",
            ]
            if x
        )
        console.print(f"[bold]{i}. {escape(h.title)}[/]  [dim]{h.type}[/]")
        console.print(f"   [dim]{escape(meta)}[/]")
        if h.url:
            console.print(f"   [blue]{escape(h.url)}[/]")
        excerpt = h.excerpt if full else (h.excerpt[:280] + ("…" if len(h.excerpt) > 280 else ""))
        console.print(f"   {escape(excerpt)}\n")

    if result.linked:
        console.print("[bold]linked:[/]")
        for nb in result.linked:
            url = nb.get("url") or ""
            console.print(
                f"   [dim]{escape(str(nb.get('type', '')))}[/] {escape(str(nb.get('title', '')))}"
                + (f"  [blue]{escape(url)}[/]" if url else "")
            )


# --------------------------------------------------------------------------- #
# mcp
# --------------------------------------------------------------------------- #
@app.command()
def mcp() -> None:
    """Run the MCP server (stdio transport) for Claude Code / Claude Desktop."""
    # FastMCP checks PyPI for a newer release while printing its banner. main() turns
    # the banner off, which skips the check; these keep any other path quiet and
    # offline too, unless set explicitly. fastmcp reads them when imported below.
    os.environ.setdefault("FASTMCP_CHECK_FOR_UPDATES", "off")
    os.environ.setdefault("FASTMCP_SHOW_SERVER_BANNER", "false")
    # stdout is the JSON-RPC wire. Anything printed while the server imports goes to
    # stderr; stdout is restored before serving, because the SDK finds the wire
    # through sys.stdout. Never reassign it around main(): JSON-RPC would go to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        from sift.mcp_server import main
    # Line-buffers stdout (a stray print flushes into the fd stdio diverted, not onto
    # the wire at exit), keeps logging on stderr, serves with no banner, and warms
    # the models after `initialize` (SIFT_MCP_WARMUP, SIFT_MCP_AUTO_SYNC).
    main()


# --------------------------------------------------------------------------- #
# ingest subcommands
# --------------------------------------------------------------------------- #
def _report(name: str, res) -> None:
    """One summary line per source; exit 1 when notes failed to save or index."""
    clean = res.complete and not (res.errors or res.id_conflicts)
    console.print(f"[{'green' if clean else 'yellow'}]{name}[/]: {escape(res.summary())}")
    if res.id_conflicts:
        err.print(
            f"[yellow]{res.id_conflicts} note(s) not saved:[/] their id belongs to a different "
            "document (see the warnings above)"
        )
    if res.errors:
        raise typer.Exit(1)


@ingest_app.command("kev")
def ingest_kev() -> None:
    """CISA Known Exploited Vulnerabilities catalog (merged into existing CVE notes)."""
    from sift.ingest.base import run_source
    from sift.ingest.kev import source

    with _bar("ingest kev") as (progress, task):
        res = run_source("kev", source(), on_progress=_titled(progress, task, "kev"))
    _report("kev", res)


@ingest_app.command("nvd")
def ingest_nvd(
    since: int = typer.Option(0, help="Start year (default: two years ago)."),
    cwe: str | None = typer.Option(None, help="Comma-separated CWEs; default = web-app set."),
    max: int = typer.Option(0, help="Stop after N notes (0 = no cap)."),
) -> None:
    """Recent CVEs from NVD, filtered to web-app CWEs (merged into existing CVE notes)."""
    from sift.ingest.base import run_source
    from sift.ingest.nvd import source

    cwes = [c.strip().upper() for c in cwe.split(",")] if cwe else None
    with _bar("ingest nvd") as (progress, task):
        res = run_source(
            "nvd",
            source(since_year=since or None, cwes=cwes, max_notes=max or None),
            on_progress=_titled(progress, task, "nvd"),
        )
    _report("nvd", res)


@ingest_app.command("epss")
def ingest_epss() -> None:
    """Enrich existing CVE notes with FIRST EPSS exploit-prediction scores."""
    from sift.ingest.epss import enrich_notes

    r = enrich_notes()
    problems = r.failed_batches or r.errors
    console.print(
        f"[{'yellow' if problems else 'green'}]epss[/]: {r.changed} changed, "
        f"{r.unchanged} unchanged, {r.scored}/{r.notes} CVE notes scored, "
        f"{r.indexed_chunks} chunks"
        + (f", {r.failed_batches} failed API batches" if r.failed_batches else "")
        + (f", {r.errors} errors" if r.errors else "")
    )
    if r.failed_batches:
        err.print("[yellow]CVEs in failed batches keep their old scores; re-run later.[/]")
    if problems:
        raise typer.Exit(1)


@ingest_app.command("h1-public")
def ingest_h1_public(
    limit: int = typer.Option(0, help="Max reports considered (0 = all ~12.6k)."),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-process reports already in the vault (default: skip them)."
    ),
) -> None:
    """Public HackerOne disclosed reports (Hugging Face dataset)."""
    from sift.ingest.base import run_source
    from sift.ingest.h1_public import source

    with _bar("ingest h1-public") as (progress, task):
        res = run_source(
            "h1-public",
            source(limit=limit or None, refresh=refresh),
            on_progress=_titled(progress, task, "h1-public"),
        )
    _report("h1-public", res)


@ingest_app.command("h1-mine")
def ingest_h1_mine(
    hacktivity: bool = typer.Option(False, help="Also pull the public hacktivity feed."),
    query: str | None = typer.Option(None, help="Lucene query for hacktivity."),
    limit: int = typer.Option(500, help="Max hacktivity reports considered."),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-process hacktivity reports already in the vault."
    ),
) -> None:
    """Your own HackerOne reports (resolved + dupes) via the API token."""
    from sift.ingest.base import run_source
    from sift.ingest.h1_api import hacktivity as hacktivity_source
    from sift.ingest.h1_api import my_reports

    res = run_source("h1-mine", my_reports())
    failed = False
    try:
        _report("h1-mine", res)
    except typer.Exit:
        failed = True

    if hacktivity:
        res2 = run_source(
            "h1-hacktivity", hacktivity_source(query=query, limit=limit, refresh=refresh)
        )
        _report("h1-hacktivity", res2)
    if failed:
        raise typer.Exit(1)


@ingest_app.command("research")
def ingest_research(
    fetch_body: bool = typer.Option(
        False, "--fetch-body", help="Also fetch each article page for full text (slower)."
    ),
    limit: int = typer.Option(0, help="Max new items (0 = all)."),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Re-fetch items already in the vault (e.g. with --fetch-body, to upgrade excerpts "
        "to full text). A refresh never shortens a stored body.",
    ),
    since: str = typer.Option(
        None,
        "--since",
        help="YYYY-MM-DD. Default: SIFT_MODEL_CUTOFF. Pass an older date with --refresh to "
        "re-fetch pre-cutoff items.",
    ),
) -> None:
    """Recent security research from RSS feeds (PortSwigger + SIFT_RESEARCH_FEEDS)."""
    from sift.ingest.base import run_source
    from sift.ingest.research import source

    horizon = _parse_since(since)
    with _bar("ingest research") as (progress, task):
        res = run_source(
            "research",
            source(limit=limit or None, fetch_body=fetch_body, refresh=refresh, since=horizon),
            on_progress=_titled(progress, task, "research"),
        )
    _report("research", res)


@ingest_app.command("writeups")
def ingest_writeups(
    since_year: int = typer.Option(
        2024, "--since-year", help="Skip writeups published before this year."
    ),
    since: str = typer.Option(
        None, "--since", help="YYYY-MM-DD; overrides --since-year when given."
    ),
    limit: int = typer.Option(0, help="Max new items (0 = all)."),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-fetch writeups already in the vault (default: skip them)."
    ),
) -> None:
    """PentesterLand's curated bug bounty writeup index (~6,400 entries)."""
    from sift.ingest.base import run_source
    from sift.ingest.writeups import source

    horizon = _parse_since(since)
    with _bar("ingest writeups") as (progress, task):
        res = run_source(
            "writeups",
            source(limit=limit or None, since_year=since_year, refresh=refresh, since=horizon),
            on_progress=_titled(progress, task, "writeups"),
        )
    _report("writeups", res)


@ingest_app.command("top10")
def ingest_top10(
    years: str = typer.Option(
        "2021,2022,2023,2024,2025", "--years", help="Comma-separated nomination years."
    ),
    limit: int = typer.Option(0, help="Max new items (0 = all)."),
    delay: float = typer.Option(1.0, "--delay", help="Seconds between third-party fetches."),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-fetch articles already stored as top10 notes."
    ),
) -> None:
    """Research linked from PortSwigger's Top 10 Web Hacking Techniques pages."""
    from sift.ingest.base import run_source
    from sift.ingest.top10 import source

    try:
        yrs = tuple(int(y.strip()) for y in years.split(",") if y.strip())
    except ValueError:
        raise _fail("--years must be comma-separated integers") from None

    with _bar("ingest top10") as (progress, task):
        res = run_source(
            "top10",
            source(years=yrs, limit=limit or None, delay=delay, refresh=refresh),
            on_progress=_titled(progress, task, "top10"),
        )
    _report("top10", res)


@ingest_app.command("notes")
def ingest_notes() -> None:
    """Backfill missing frontmatter in hand-written notes, then index the vault
    incrementally (only new or changed notes are embedded)."""
    from sift.ingest.local_notes import backfill_and_index
    from sift.pipeline import count_notes

    total = count_notes()
    try:
        with _bar(f"indexing {total} notes", total=total) as (progress, task):
            res = backfill_and_index(on_progress=lambda done: progress.update(task, completed=done))
            if res.stats is not None:
                walked = getattr(res.stats, "walked", None)
                if walked is not None:
                    progress.update(task, total=walked, completed=walked)
    except (RuntimeError, ValueError) as exc:
        raise _fail(str(exc)) from None
    console.print(f"[green]notes[/]: {res.fixed} frontmatter backfilled, {res.indexed} indexed")
    if res.stats is not None:
        _print_reindex_stats(res.stats, label="index")
    if res.problems:
        err.print(f"[yellow]{len(res.problems)} file(s) left untouched:[/]")
        for path, why in res.problems[:10]:
            err.print(f"  {escape(str(path))}: {escape(str(why))}")
        if len(res.problems) > 10:
            err.print(f"  ... and {len(res.problems) - 10} more")
        raise typer.Exit(1)


@ingest_app.command("url")
def ingest_url(
    url: str = typer.Argument(help="http(s) URL of one article."),
    program: str | None = typer.Option(None, "--program", help="Program / vendor it concerns."),
    tag: Annotated[list[str] | None, typer.Option("--tag", help="Extra tag (repeatable).")] = None,
    force: bool = typer.Option(
        False, "--force", help="Capture even if it predates the model cutoff or was pruned."
    ),
) -> None:
    """Capture one fresh article verbatim (the MCP capture_url tool's path).

    Refuses private and internal hosts (on every redirect hop), pages that don't
    extract to an article, and articles older than SIFT_MODEL_CUTOFF unless --force.
    """
    from fastmcp.exceptions import ToolError

    from sift.mcp_server import capture_url

    try:
        out = capture_url(url, program=program, tags=list(tag) if tag else None, force=force)
    except ToolError as exc:
        raise _fail(str(exc)) from None
    if out.get("saved"):
        console.print(
            f"[green]saved[/] {escape(str(out.get('title', '')))}  [dim]id {out.get('note_id')}, "
            f"{out.get('chunks_indexed', 0)} chunks[/]"
        )
        console.print(f"   {escape(str(out.get('path', '')))}")
        return
    if out.get("existing"):
        console.print(
            f"already in the vault: {escape(str(out.get('title', '')))}  "
            f"[dim]id {out.get('note_id')}[/]"
        )
        return
    reason = str(out.get("reason") or "not saved")
    if out.get("hint"):
        reason += f" ({out['hint']})"
    raise _fail(f"not saved: {reason}")


# --------------------------------------------------------------------------- #
# ingest: Google / Chromium
#
# All three default their horizon to SIFT_MODEL_CUTOFF. These surfaces are heavily
# represented in training data, so an unbounded backfill costs retrieval tokens to
# store things the reasoning model can already recite; the window is where the
# information is.
# --------------------------------------------------------------------------- #


def _parse_since(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise _fail(f"--since must be YYYY-MM-DD, got {raw!r}") from None


@ingest_app.command("chromium-docs")
def ingest_chromium_docs(
    since: str = typer.Option(None, "--since", help="YYYY-MM-DD. Default: SIFT_MODEL_CUTOFF."),
    all_docs: bool = typer.Option(
        False, "--all", help="Every doc in scope, not just those changed since the cutoff."
    ),
    limit: int = typer.Option(0, help="Max docs (0 = all)."),
) -> None:
    """Chromium in-tree security/IPC docs the vendor changed since the cutoff."""
    from sift.ingest.base import run_source
    from sift.ingest.chromium_docs import source

    with _bar("ingest chromium-docs") as (progress, task):
        res = run_source(
            "chromium-docs",
            source(since=_parse_since(since), all_docs=all_docs, limit=limit or None),
            on_progress=_titled(progress, task, "chromium-docs"),
        )
    _report("chromium-docs", res)


@ingest_app.command("chromium-fixes")
def ingest_chromium_fixes(
    since: str = typer.Option(None, "--since", help="YYYY-MM-DD. Default: SIFT_MODEL_CUTOFF."),
    classes: str = typer.Option(
        "",
        "--classes",
        help="Comma-separated: memory-safety,boundary-enforcement,threat-model,lifetime.",
    ),
    paths: str = typer.Option(
        "", "--paths", help="Comma-separated repo paths to mine instead of the defaults."
    ),
    limit: int = typer.Option(0, help="Max commits (0 = all)."),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-process fix commits already in the vault."
    ),
) -> None:
    """Chromium security-fix commits — the post-cutoff mechanism corpus."""
    from sift.ingest.base import run_source
    from sift.ingest.chromium_fixes import CLASS_PATTERNS, DEFAULT_PATHS, source

    wanted = tuple(c.strip() for c in classes.split(",") if c.strip())
    unknown = [c for c in wanted if c not in CLASS_PATTERNS]
    if unknown:
        raise _fail(f"unknown class(es) {unknown}; known: {list(CLASS_PATTERNS)}")
    mine = tuple(p.strip() for p in paths.split(",") if p.strip()) or DEFAULT_PATHS

    with _bar("ingest chromium-fixes") as (progress, task):
        res = run_source(
            "chromium-fixes",
            source(
                since=_parse_since(since),
                paths=mine,
                classes=wanted,
                limit=limit or None,
                refresh=refresh,
            ),
            on_progress=_titled(progress, task, "chromium-fixes"),
        )
    _report("chromium-fixes", res)


@ingest_app.command("chrome-releases")
def ingest_chrome_releases(
    since: str = typer.Option(None, "--since", help="YYYY-MM-DD. Default: SIFT_MODEL_CUTOFF."),
    limit: int = typer.Option(0, help="Max release posts (0 = all in window)."),
    no_ledger: bool = typer.Option(
        False, "--no-ledger", help="Skip the aggregate reward ledger note."
    ),
) -> None:
    """Chrome release security tables + the reward ledger (component, class, payout)."""
    from sift.ingest.base import run_source
    from sift.ingest.chrome_releases import source

    with _bar("ingest chrome-releases") as (progress, task):
        res = run_source(
            "chrome-releases",
            source(since=_parse_since(since), limit=limit or None, ledger=not no_ledger),
            on_progress=_titled(progress, task, "chrome-releases"),
        )
    _report("chrome-releases", res)


@ingest_app.command("google")
def ingest_google(
    since: str = typer.Option(None, "--since", help="YYYY-MM-DD. Default: SIFT_MODEL_CUTOFF."),
) -> None:
    """Run every Google/Chromium source: docs, fix commits, release tables.

    Each runs even if an earlier one fails (chrome-releases needs no checkout); the
    command exits 1 if any failed.
    """
    horizon = _parse_since(since) or get_settings().model_cutoff  # a bad --since exits here
    console.print(f"[bold]Google/Chromium ingest[/] — horizon {horizon.isoformat()}")
    # Called as plain functions, so every option is passed explicitly: an omitted one
    # would be its typer.Option default object, which is truthy.
    steps = (
        ("chromium-docs", lambda: ingest_chromium_docs(since=since, all_docs=False, limit=0)),
        (
            "chromium-fixes",
            lambda: ingest_chromium_fixes(
                since=since, classes="", paths="", limit=0, refresh=False
            ),
        ),
        (
            "chrome-releases",
            lambda: ingest_chrome_releases(since=since, limit=0, no_ledger=False),
        ),
    )
    failed: list[str] = []
    for name, run in steps:
        try:
            run()
        except typer.Exit as exc:  # subclasses RuntimeError: must come before Exception
            if exc.exit_code:
                failed.append(name)
        except Exception as exc:  # noqa: BLE001 - one broken source must not cost the others
            failed.append(name)
            err.print(f"[red]{name} failed:[/] {escape(str(exc))}")
    if failed:
        err.print(f"[yellow]failed: {', '.join(failed)}[/]")
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# distill
# --------------------------------------------------------------------------- #
@distill_app.command("export")
def distill_export(
    out: str = typer.Option("candidates.jsonl", "-o", "--out", help="Where to write candidates."),
    note_type: str = typer.Option("writeup", "--type", help="Note type to pull candidates from."),
    limit: int = typer.Option(0, help="Max candidates (0 = all)."),
    all_notes: bool = typer.Option(False, "--all", help="Include already-judged material."),
) -> None:
    """Dump ungated candidates for the model to judge in-session (no API key needed)."""
    from sift.distill.manual import collect, write_candidates

    cands = collect(note_type, limit=limit or None, skip_gated=not all_notes)
    if not cands:
        console.print("[yellow]nothing to judge[/] — all candidates already gated")
        raise typer.Exit(0)
    n = write_candidates(cands, Path(out))
    console.print(f"[green]exported[/] {n} candidates -> {out}")
    console.print(
        "[dim]Rows with truncated=true carry only an excerpt: read the note at `path` before "
        "writing body_md. Apply with `sift distill apply VERDICTS --candidates "
        f"{escape(out)}`.[/]"
    )


@distill_app.command("apply")
def distill_apply(
    verdicts: str = typer.Argument(help="JSONL of verdicts."),
    candidates: str = typer.Option(
        "candidates.jsonl",
        "--candidates",
        help="The export the verdicts were judged from (default: export's default --out).",
    ),
    note_type: str = typer.Option(
        "writeup",
        "--type",
        help="Only when the candidates file is gone: the note type the export came from.",
    ),
) -> None:
    """Turn verdicts into technique notes; log the drops."""
    from sift.distill.manual import apply_verdicts, collect, load_candidates

    path = Path(verdicts)
    if not path.exists():
        raise _fail(f"no such file: {verdicts}")
    cpath = Path(candidates)
    if cpath.is_file():
        cands = load_candidates(cpath)
        err.print(
            f"[dim]matching verdicts against {escape(str(cpath))} ({len(cands)} candidates)[/]"
        )
    else:
        err.print(
            f"[dim]{escape(str(cpath))} not found; matching against the vault's "
            f"{escape(note_type)} notes[/]"
        )
        cands = collect(note_type, skip_gated=False)
    res = apply_verdicts(path, cands)
    line = (
        f"[green]distill[/]: {res['kept']} kept, {res['dropped']} dropped, "
        f"{res['skipped']} skipped, {res['chunks_indexed']} chunks"
    )
    if res.get("duplicates"):
        line += f", {res['duplicates']} drops already logged"
    if res.get("collisions"):
        line += f", {res['collisions']} id clash(es) saved under a url-hashed id"
    console.print(line)
    problems = res.get("problems") or []
    for p in problems:
        err.print(f"  [yellow]![/] {escape(str(p))}")
    if problems:
        raise typer.Exit(1)


@distill_app.command("eval")
def distill_eval(
    designs: str = typer.Option(
        "self-report,behavioural", "--designs", help="Comma-separated: self-report, behavioural."
    ),
    limit: int = typer.Option(0, help="Only score the first N labelled candidates (0 = all)."),
) -> None:
    """Score gate designs against the hand-labelled verdicts. Costs a few cents."""
    from sift.distill.behavioral import judge_behavioral
    from sift.distill.evaluate import format_report, labelled_candidates, load_labels, score_gate
    from sift.distill.gate import GateError, _client, judge

    available = {"self-report": judge, "behavioural": judge_behavioral}
    picked = [d.strip() for d in designs.split(",") if d.strip()]
    unknown = [d for d in picked if d not in available]
    if unknown:
        raise _fail(f"unknown design(s): {', '.join(unknown)} — pick from {list(available)}")

    labels = load_labels()
    pairs = labelled_candidates(labels, get_settings().resolved_vault())
    if not pairs:
        raise _fail("no labelled notes found in the vault — are the writeups still ingested?")
    if limit:
        pairs = pairs[:limit]

    console.print(f"scoring {len(pairs)} labelled candidates across {len(picked)} design(s)...")
    try:
        client = _client()
    except GateError as exc:
        err.print(f"[yellow]{escape(str(exc))}[/]")
        raise typer.Exit(1) from None

    done = {"n": 0}
    total = len(pairs) * len(picked)

    def progress(name, cand, verdict, human):
        done["n"] += 1
        mark = (
            "?"
            if verdict is None
            else ("ok" if (("keep" if verdict.keep else "drop") == human) else "XX")
        )
        console.print(f"  [{done['n']}/{total}] {mark} {name:12} {escape(cand.title[:52])}")

    try:
        scores = [
            score_gate(d, available[d], pairs, client=client, on_result=progress) for d in picked
        ]
    except GateError as exc:  # revoked key, billing, unknown model, repeated 400s
        err.print(f"[yellow]{escape(str(exc))}[/]")
        raise typer.Exit(1) from None
    console.print(format_report(scores))


@distill_app.command("rejects")
def distill_rejects(limit: int = typer.Option(20, "-n", help="Show the most recent N.")) -> None:
    """Review what the gate dropped — the evidence base for retuning strictness."""
    from sift.distill.rejects import load_rejects

    rows = load_rejects(limit=limit)
    if not rows:
        console.print("[yellow]no rejects logged yet[/]")
        raise typer.Exit(0)
    table = Table("title", "source", "why the model said it already knew it")
    for r in rows:
        table.add_row(
            escape(str(r.get("title", ""))[:60]),
            escape(str(r.get("source", ""))),
            escape(str(r.get("already_known", ""))[:80]),
        )
    console.print(table)


# --------------------------------------------------------------------------- #
# trash (soft deletes by the MCP forget_note tool)
# --------------------------------------------------------------------------- #
@trash_app.command("list")
def trash_list() -> None:
    """List the notes in the vault's .trash folder."""
    from sift.vault.notes import TRASH_DIR, describe_error, load_note

    vault = get_settings().resolved_vault()
    root = vault / TRASH_DIR
    files = sorted(root.rglob("*.md")) if root.is_dir() else []
    if not files:
        console.print("trash is empty")
        return
    t = Table(title=f"{len(files)} trashed note(s)")
    for col in ("id", "title", "deleted", "why", "file"):
        t.add_column(col, overflow="fold")
    for p in files:
        rel = p.relative_to(root).as_posix()
        try:
            note = load_note(p)
        except Exception as exc:  # noqa: BLE001
            t.add_row("?", f"[dim]{escape(describe_error(exc))}[/]", "", "", escape(rel))
            continue
        extra = note.meta.extra if isinstance(note.meta.extra, dict) else {}
        t.add_row(
            escape(note.meta.id),
            escape(note.meta.title[:60]),
            escape(str(extra.get("deleted", ""))),
            escape(str(extra.get("deleted_reason", ""))[:60]),
            escape(rel),
        )
    console.print(t)
    console.print(
        "[dim]Restore one with `sift trash restore NOTE_ID`; delete the folder's files by hand "
        "once you are sure.[/]"
    )


@trash_app.command("restore")
def trash_restore(note_id: str = typer.Argument(help="The trashed note's id.")) -> None:
    """Move a trashed note back into the vault and clear its tombstone.

    Run `sift reindex` afterwards to make it searchable again.
    """
    from sift.tombstones import remove_tombstones
    from sift.vault.notes import TRASH_DIR, load_note, locate_note, write_lock, write_note

    vault = get_settings().resolved_vault()
    root = vault / TRASH_DIR
    restored: list[Path] = []
    conflicts: list[str] = []
    with write_lock(vault):
        matches = []
        for p in sorted(root.rglob("*.md")) if root.is_dir() else []:
            try:
                note = load_note(p)
            except Exception:  # noqa: BLE001 - not the one we want, or unreadable
                continue
            if note.meta.id == note_id:
                matches.append(note)
        if not matches:
            raise _fail(f"no trashed note has id {note_id!r} (see `sift trash list`)")
        if locate_note(vault, note_id, max_age=0.0) is not None:
            raise _fail(
                f"a note with id {note_id!r} is in the vault again; restoring would make a "
                "second file for one id. Compare the two and move the file by hand."
            )
        for note in matches:
            src = Path(note.path)
            rel = src.relative_to(root)
            target = vault / rel
            if target.exists():
                conflicts.append(rel.as_posix())
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, target)  # never replaces: the target was checked under the lock
            extra = dict(note.meta.extra) if isinstance(note.meta.extra, dict) else {}
            stamped = extra.pop("deleted", None) is not None
            stamped = extra.pop("deleted_reason", None) is not None or stamped
            if stamped:
                note.meta.extra = extra
                note.path = target
                write_note(vault, note, existing=target, stamp=False, rename=False)
            restored.append(target)
        urls = sorted({n.meta.url for n in matches if n.meta.url})
        cleared = remove_tombstones(ids=[note_id], urls=urls) if restored else 0
    for p in restored:
        console.print(f"[green]restored[/] {escape(p.relative_to(vault).as_posix())}")
    if cleared:
        console.print(f"  {cleared} tombstone(s) cleared")
    for c in conflicts:
        err.print(f"[yellow]left in .trash:[/] {escape(c)} (the vault already has that file)")
    if restored:
        console.print("next: [bold]sift reindex[/] to make it searchable again")
    if conflicts:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
