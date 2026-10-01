"""sift command-line interface."""

from __future__ import annotations

import contextlib
import sys
from datetime import date
from pathlib import Path

# Windows terminals default to cp1252 and mangle em-dashes / box chars in note text.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import typer
from rich.console import Console
from rich.table import Table

from sift.config import get_settings
from sift.vault.schema import NOTE_TYPES

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
ingest_app = typer.Typer(no_args_is_help=True, help="Pull external data into the vault.")
app.add_typer(ingest_app, name="ingest")
distill_app = typer.Typer(no_args_is_help=True, help="Gate source material into technique notes.")
app.add_typer(distill_app, name="distill")
console = Console()


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


# --------------------------------------------------------------------------- #
# init / status
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

    env = get_settings().model_config["env_file"]
    if not Path(env).exists():
        example = Path(env).with_name(".env.example")
        if example.exists():
            Path(env).write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
            console.print(f"[green]wrote[/] {env} (from .env.example — fill in your keys)")

    console.print(f"[green]vault ready[/] at {vault}")
    console.print('next: [bold]sift ingest kev[/] then [bold]sift search "..."[/]')


@app.command()
def status() -> None:
    """Show note counts, index size, and last ingest times."""
    from sift.index.store import Store
    from sift.ingest.base import load_state
    from sift.vault.notes import iter_notes

    s = get_settings()
    vault = s.resolved_vault()

    counts: dict[str, int] = {}
    for note in iter_notes(vault):
        counts[note.meta.type] = counts.get(note.meta.type, 0) + 1

    t = Table(title="vault")
    t.add_column("type")
    t.add_column("notes", justify="right")
    for k in NOTE_TYPES:
        t.add_row(k, str(counts.get(k, 0)))
    t.add_row("[bold]total", f"[bold]{sum(counts.values())}")
    console.print(t)

    console.print(f"index chunks: [bold]{Store().count()}[/]  ({s.resolved_db()})")
    console.print(f"embeddings: {s.embed_model} on {s.embed_device}   rerank: {s.rerank}")

    state = load_state()
    if state:
        st = Table(title="last ingest")
        st.add_column("source")
        st.add_column("when")
        st.add_column("written", justify="right")
        for src, info in sorted(state.items()):
            st.add_row(src, info.get("last_run", "?"), str(info.get("written", "?")))
        console.print(st)


# --------------------------------------------------------------------------- #
# reindex / search
# --------------------------------------------------------------------------- #
@app.command()
def reindex(force: bool = typer.Option(False, help="Drop and rebuild the whole index.")) -> None:
    """Rebuild the LanceDB index from the markdown vault."""
    from sift.pipeline import count_notes
    from sift.pipeline import reindex as _reindex

    total = count_notes()
    with _bar(f"reindexing {total} notes", total=total) as (progress, task):
        stats = _reindex(
            force=force, on_progress=lambda done: progress.update(task, completed=done)
        )
    unchanged = f"  [dim]({stats.unchanged} unchanged, skipped)[/]" if stats.unchanged else ""
    console.print(
        f"[green]done[/] {stats.notes} notes, {stats.chunks} chunks "
        f"({', '.join(f'{k}:{v}' for k, v in sorted(stats.by_type.items()))}){unchanged}"
    )


@app.command()
def prune(
    keep_since: int = typer.Option(2025, help="Keep reports/CVEs created in this year or later."),
    report_quality_bar: int = typer.Option(
        62, help="Older reports are kept only if bountied AND scoring at least this."
    ),
    yes: bool = typer.Option(False, "--yes", help="Actually delete (default: dry run)."),
) -> None:
    """Drop bulk-ingested notes the reasoning model already knows; keep the rest.

    Default is a dry run — shows what would go. Re-run with --yes to delete and reindex.
    """
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
        t.add_row("[green]keep", r, str(n))
    for r, n in drop_reasons.most_common():
        t.add_row("[red]drop", r, str(n))
    t.add_row("[bold]total", "", f"[bold]{len(keep) + len(drop)}")
    console.print(t)
    console.print(f"[green]keep {len(keep)}[/]   [red]drop {len(drop)}[/]")

    if drop:
        console.print("\n[dim]sample drops:[/]")
        for note in drop[:10]:
            console.print(f"  [dim]{note.meta.type}[/] {note.meta.title[:80]}")

    if not yes:
        console.print("\n[yellow]dry run[/] — re-run with [bold]--yes[/] to delete and reindex.")
        return

    removed = 0
    for note in drop:
        if note.path and note.path.exists():
            note.path.unlink()
            removed += 1
    console.print(f"[red]deleted {removed} notes[/]. Rebuilding index…")
    from sift.pipeline import reindex as _reindex

    stats = _reindex(force=True)
    console.print(f"[green]done[/] {stats.notes} notes, {stats.chunks} chunks")


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
    result = _search(
        query, k=k, filters=filters or None, expand_links=links, min_quality=min_quality
    )

    if not result.hits:
        console.print("[yellow]no matches[/] — is the index built? try `sift reindex`")
        raise typer.Exit(1)

    from rich.markup import escape

    for i, h in enumerate(result.hits, 1):
        meta = "  ".join(
            x
            for x in [
                h.severity and f"sev:{h.severity}",
                h.program and f"@{h.program}",
                f"q:{h.quality}",
                f"score {h.score:.3f}",
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
            console.print(
                f"   [dim]{nb['type']}[/] {escape(nb['title'])}  [blue]{escape(nb['url'])}[/]"
            )


# --------------------------------------------------------------------------- #
# mcp
# --------------------------------------------------------------------------- #
@app.command()
def mcp() -> None:
    """Run the MCP server (stdio transport) for Claude Code / Claude Desktop."""
    from sift.mcp_server import mcp as server

    server.run()


# --------------------------------------------------------------------------- #
# ingest subcommands
# --------------------------------------------------------------------------- #
@ingest_app.command("kev")
def ingest_kev() -> None:
    """CISA Known Exploited Vulnerabilities catalog."""
    from sift.ingest.base import run_source
    from sift.ingest.kev import source

    res = run_source("kev", source())
    console.print(
        f"[green]kev[/]: {res.written} notes, {res.indexed_chunks} chunks, {res.errors} errors"
    )


@ingest_app.command("nvd")
def ingest_nvd(
    since: int = typer.Option(0, help="Start year (default: two years ago)."),
    cwe: str | None = typer.Option(None, help="Comma-separated CWEs; default = web-app set."),
    max: int = typer.Option(0, help="Stop after N notes (0 = no cap)."),
) -> None:
    """Recent CVEs from NVD, filtered to web-app CWEs."""
    from sift.ingest.base import run_source
    from sift.ingest.nvd import source

    cwes = [c.strip().upper() for c in cwe.split(",")] if cwe else None
    res = run_source(
        "nvd",
        source(since_year=since or None, cwes=cwes, max_notes=max or None),
    )
    console.print(
        f"[green]nvd[/]: {res.written} notes, {res.indexed_chunks} chunks, {res.errors} errors"
    )


@ingest_app.command("epss")
def ingest_epss() -> None:
    """Enrich existing CVE notes with FIRST EPSS exploit-prediction scores."""
    from sift.ingest.epss import enrich

    n = enrich()
    console.print(f"[green]epss[/]: enriched {n} CVE notes")


@ingest_app.command("h1-public")
def ingest_h1_public(limit: int = typer.Option(0, help="Max reports (0 = all ~12.6k).")) -> None:
    """Public HackerOne disclosed reports (Hugging Face dataset)."""
    from sift.ingest.base import run_source
    from sift.ingest.h1_public import source

    res = run_source("h1-public", source(limit=limit or None))
    console.print(
        f"[green]h1-public[/]: {res.written} notes, {res.indexed_chunks} chunks, {res.errors} errors"
    )


@ingest_app.command("h1-mine")
def ingest_h1_mine(
    hacktivity: bool = typer.Option(False, help="Also pull the public hacktivity feed."),
    query: str | None = typer.Option(None, help="Lucene query for hacktivity."),
    limit: int = typer.Option(500, help="Max hacktivity reports."),
) -> None:
    """Your own HackerOne reports (resolved + dupes) via the API token."""
    from sift.ingest.base import run_source
    from sift.ingest.h1_api import hacktivity as hacktivity_source
    from sift.ingest.h1_api import my_reports

    res = run_source("h1-mine", my_reports())
    console.print(
        f"[green]h1-mine[/]: {res.written} notes, {res.indexed_chunks} chunks, {res.errors} errors"
    )

    if hacktivity:
        res2 = run_source("h1-hacktivity", hacktivity_source(query=query, limit=limit))
        console.print(f"[green]h1-hacktivity[/]: {res2.written} notes, {res2.errors} errors")


@ingest_app.command("research")
def ingest_research(
    fetch_body: bool = typer.Option(
        False, "--fetch-body", help="Also fetch each article page for full text (slower)."
    ),
    limit: int = typer.Option(0, help="Max new items (0 = all)."),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-fetch items already in the vault (e.g. to upgrade excerpts to full text)."
    ),
) -> None:
    """Recent security research from RSS feeds (PortSwigger + SIFT_RESEARCH_FEEDS)."""
    from sift.ingest.base import run_source
    from sift.ingest.research import source

    with _bar("ingest research") as (progress, task):
        res = run_source(
            "research",
            source(limit=limit or None, fetch_body=fetch_body, refresh=refresh),
            on_progress=lambda n, title: progress.update(task, completed=n, description=f"ingest research: {title[:44]}"),
        )
    console.print(
        f"[green]research[/]: {res.written} new notes, {res.indexed_chunks} chunks, "
        f"{res.errors} errors"
    )


@ingest_app.command("writeups")
def ingest_writeups(
    since_year: int = typer.Option(2024, "--since-year", help="Skip writeups published before this year."),
    limit: int = typer.Option(0, help="Max new items (0 = all)."),
) -> None:
    """PentesterLand's curated bug bounty writeup index (~6,400 entries)."""
    from sift.ingest.base import run_source
    from sift.ingest.writeups import source

    with _bar("ingest writeups") as (progress, task):
        res = run_source(
            "writeups",
            source(limit=limit or None, since_year=since_year),
            on_progress=lambda n, title: progress.update(task, completed=n, description=f"ingest writeups: {title[:44]}"),
        )
    console.print(
        f"[green]writeups[/]: {res.written} new notes, {res.indexed_chunks} chunks, {res.errors} errors"
    )


@ingest_app.command("top10")
def ingest_top10(
    years: str = typer.Option("2021,2022,2023,2024,2025", "--years", help="Comma-separated nomination years."),
    limit: int = typer.Option(0, help="Max new items (0 = all)."),
    delay: float = typer.Option(1.0, "--delay", help="Seconds between third-party fetches."),
) -> None:
    """Research linked from PortSwigger's Top 10 Web Hacking Techniques pages."""
    from sift.ingest.base import run_source
    from sift.ingest.top10 import source

    try:
        yrs = tuple(int(y.strip()) for y in years.split(",") if y.strip())
    except ValueError:
        console.print("[yellow]--years must be comma-separated integers[/]")
        raise typer.Exit(1) from None

    with _bar("ingest top10") as (progress, task):
        res = run_source(
            "top10",
            source(years=yrs, limit=limit or None, delay=delay),
            on_progress=lambda n, title: progress.update(task, completed=n, description=f"ingest top10: {title[:44]}"),
        )
    console.print(
        f"[green]top10[/]: {res.written} new notes, {res.indexed_chunks} chunks, {res.errors} errors"
    )


@ingest_app.command("notes")
def ingest_notes() -> None:
    """Index hand-authored markdown; backfill missing frontmatter."""
    from sift.ingest.local_notes import backfill_and_index

    fixed, indexed = backfill_and_index()
    console.print(f"[green]notes[/]: {indexed} indexed, {fixed} frontmatter backfilled")


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
        console.print(f"[yellow]--since must be YYYY-MM-DD, got {raw!r}[/]")
        raise typer.Exit(1) from None


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
            on_progress=lambda n, title: progress.update(
                task, completed=n, description=f"ingest chromium-docs: {title[:44]}"
            ),
        )
    console.print(
        f"[green]chromium-docs[/]: {res.written} new notes, {res.indexed_chunks} chunks, "
        f"{res.errors} errors"
    )


@ingest_app.command("chromium-fixes")
def ingest_chromium_fixes(
    since: str = typer.Option(None, "--since", help="YYYY-MM-DD. Default: SIFT_MODEL_CUTOFF."),
    classes: str = typer.Option(
        "", "--classes", help="Comma-separated: memory-safety,boundary-enforcement,threat-model,lifetime."
    ),
    paths: str = typer.Option("", "--paths", help="Comma-separated repo paths to mine instead of the defaults."),
    limit: int = typer.Option(0, help="Max commits (0 = all)."),
) -> None:
    """Chromium security-fix commits — the post-cutoff mechanism corpus."""
    from sift.ingest.base import run_source
    from sift.ingest.chromium_fixes import CLASS_PATTERNS, DEFAULT_PATHS, source

    wanted = tuple(c.strip() for c in classes.split(",") if c.strip())
    unknown = [c for c in wanted if c not in CLASS_PATTERNS]
    if unknown:
        console.print(f"[yellow]unknown class(es) {unknown}; known: {list(CLASS_PATTERNS)}[/]")
        raise typer.Exit(1)
    mine = tuple(p.strip() for p in paths.split(",") if p.strip()) or DEFAULT_PATHS

    with _bar("ingest chromium-fixes") as (progress, task):
        res = run_source(
            "chromium-fixes",
            source(since=_parse_since(since), paths=mine, classes=wanted, limit=limit or None),
            on_progress=lambda n, title: progress.update(
                task, completed=n, description=f"ingest chromium-fixes: {title[:44]}"
            ),
        )
    console.print(
        f"[green]chromium-fixes[/]: {res.written} new notes, {res.indexed_chunks} chunks, "
        f"{res.errors} errors"
    )


@ingest_app.command("chrome-releases")
def ingest_chrome_releases(
    since: str = typer.Option(None, "--since", help="YYYY-MM-DD. Default: SIFT_MODEL_CUTOFF."),
    limit: int = typer.Option(0, help="Max release posts (0 = all in window)."),
    no_ledger: bool = typer.Option(False, "--no-ledger", help="Skip the aggregate reward ledger note."),
) -> None:
    """Chrome release security tables + the reward ledger (component, class, payout)."""
    from sift.ingest.base import run_source
    from sift.ingest.chrome_releases import source

    with _bar("ingest chrome-releases") as (progress, task):
        res = run_source(
            "chrome-releases",
            source(since=_parse_since(since), limit=limit or None, ledger=not no_ledger),
            on_progress=lambda n, title: progress.update(
                task, completed=n, description=f"ingest chrome-releases: {title[:44]}"
            ),
        )
    console.print(
        f"[green]chrome-releases[/]: {res.written} new notes, {res.indexed_chunks} chunks, "
        f"{res.errors} errors"
    )


@ingest_app.command("google")
def ingest_google(
    since: str = typer.Option(None, "--since", help="YYYY-MM-DD. Default: SIFT_MODEL_CUTOFF."),
) -> None:
    """Run every Google/Chromium source: docs, fix commits, release tables."""
    horizon = _parse_since(since) or get_settings().model_cutoff
    console.print(f"[bold]Google/Chromium ingest[/] — horizon {horizon.isoformat()}")
    ingest_chromium_docs(since=since, all_docs=False, limit=0)
    ingest_chromium_fixes(since=since, classes="", paths="", limit=0)
    ingest_chrome_releases(since=since, limit=0, no_ledger=False)


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


@distill_app.command("apply")
def distill_apply(
    verdicts: str = typer.Argument(help="JSONL of verdicts."),
    note_type: str = typer.Option("writeup", "--type", help="Note type the candidates came from."),
) -> None:
    """Turn verdicts into technique notes; log the drops."""
    from sift.distill.manual import apply_verdicts, collect

    path = Path(verdicts)
    if not path.exists():
        console.print(f"[yellow]no such file:[/] {verdicts}")
        raise typer.Exit(1)
    res = apply_verdicts(path, collect(note_type, skip_gated=False))
    console.print(
        f"[green]distill[/]: {res['kept']} kept, {res['dropped']} dropped, "
        f"{res['skipped']} skipped, {res['chunks_indexed']} chunks"
    )


@distill_app.command("eval")
def distill_eval(
    designs: str = typer.Option("self-report,behavioural", "--designs", help="Comma-separated: self-report, behavioural."),
    limit: int = typer.Option(0, help="Only score the first N labelled candidates (0 = all)."),
) -> None:
    """Score gate designs against the hand-labelled verdicts. Costs a few cents."""
    from rich.markup import escape

    from sift.distill.behavioral import judge_behavioral
    from sift.distill.evaluate import format_report, labelled_candidates, load_labels, score_gate
    from sift.distill.gate import GateError, _client, judge

    available = {"self-report": judge, "behavioural": judge_behavioral}
    picked = [d.strip() for d in designs.split(",") if d.strip()]
    unknown = [d for d in picked if d not in available]
    if unknown:
        console.print(f"[yellow]unknown design(s): {', '.join(unknown)}[/] — pick from {list(available)}")
        raise typer.Exit(1)

    labels = load_labels()
    pairs = labelled_candidates(labels, get_settings().resolved_vault())
    if not pairs:
        console.print("[yellow]no labelled notes found in the vault[/] — are the writeups still ingested?")
        raise typer.Exit(1)
    if limit:
        pairs = pairs[:limit]

    console.print(f"scoring {len(pairs)} labelled candidates across {len(picked)} design(s)...")
    try:
        client = _client()
    except GateError as exc:
        console.print(f"[yellow]{exc}[/]")
        raise typer.Exit(1) from None

    done = {"n": 0}
    total = len(pairs) * len(picked)

    def progress(name, cand, verdict, human):
        done["n"] += 1
        mark = "?" if verdict is None else ("ok" if (("keep" if verdict.keep else "drop") == human) else "XX")
        console.print(f"  [{done['n']}/{total}] {mark} {name:12} {escape(cand.title[:52])}")

    scores = [score_gate(d, available[d], pairs, client=client, on_result=progress) for d in picked]
    console.print(format_report(scores))


@distill_app.command("rejects")
def distill_rejects(limit: int = typer.Option(20, "-n", help="Show the most recent N.")) -> None:
    """Review what the gate dropped — the evidence base for retuning strictness."""
    from rich.markup import escape

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


if __name__ == "__main__":
    app()
