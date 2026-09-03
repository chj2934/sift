"""sift command-line interface."""

from __future__ import annotations

import contextlib
import sys
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
console = Console()


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
    from sift.pipeline import reindex as _reindex

    console.print("reindexing…")
    stats = _reindex(force=force)
    console.print(
        f"[green]done[/] {stats.notes} notes, {stats.chunks} chunks "
        f"({', '.join(f'{k}:{v}' for k, v in sorted(stats.by_type.items()))})"
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
) -> None:
    """Recent security research from RSS feeds (PortSwigger + SIFT_RESEARCH_FEEDS)."""
    from sift.ingest.base import run_source
    from sift.ingest.research import source

    res = run_source("research", source(limit=limit or None, fetch_body=fetch_body))
    console.print(
        f"[green]research[/]: {res.written} new notes, {res.indexed_chunks} chunks, "
        f"{res.errors} errors"
    )


@ingest_app.command("notes")
def ingest_notes() -> None:
    """Index hand-authored markdown; backfill missing frontmatter."""
    from sift.ingest.local_notes import backfill_and_index

    fixed, indexed = backfill_and_index()
    console.print(f"[green]notes[/]: {indexed} indexed, {fixed} frontmatter backfilled")


if __name__ == "__main__":
    app()
