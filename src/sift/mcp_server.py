"""MCP server exposing the vault memory as tools for Claude Code / Claude Desktop.

Run with ``sift mcp`` (stdio transport). The server holds no LLM — Claude (the
host) does the reasoning; this just retrieves, reads, and persists notes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastmcp import FastMCP

from sift.config import get_settings
from sift.pipeline import get_note_by_slug, index_note
from sift.pipeline import search as _search
from sift.vault.notes import Note, iter_notes, save_note
from sift.vault.schema import NOTE_TYPES, Frontmatter

mcp = FastMCP(
    name="sift",
    instructions=(
        "Personal bug-bounty memory: disclosed reports, CVEs, techniques, targets, and the "
        "user's own findings, stored as an Obsidian-style markdown vault. Use `search_memory` "
        "before answering questions about vulnerabilities, past reports, or techniques. Use "
        "`remember` to save durable knowledge (a new technique, a finding, notes on a target). "
        "Only operate against targets the user is explicitly authorized to test."
    ),
)


@mcp.tool
def search_memory(
    query: str,
    k: int = 8,
    type: str | None = None,
    cwe: str | None = None,
    program: str | None = None,
    min_quality: int = 0,
    expand_links: bool = False,
) -> dict:
    """Hybrid (semantic + keyword) search over the memory vault.

    Args:
        query: natural-language or keyword query.
        k: number of notes to return (default 8).
        type: optional filter — one of report, cve, technique, target, finding, writeup.
        cwe: optional CWE filter, e.g. "CWE-79".
        program: optional bug bounty program / vendor filter.
        min_quality: drop hits below this 0-100 heuristic quality score (default 0 = off).
        expand_links: also return notes linked (1 hop) from the top hits.
    """
    filters = {k2: v for k2, v in {"type": type, "cwe": cwe, "program": program}.items() if v}
    res = _search(
        query, k=k, filters=filters or None, expand_links=expand_links, min_quality=min_quality
    )
    return {
        "query": query,
        "results": [
            {
                "slug": h.slug,
                "title": h.title,
                "type": h.type,
                "severity": h.severity or None,
                "program": h.program or None,
                "url": h.url or None,
                "quality": h.quality,
                "score": round(h.score, 4),
                "matched_section": h.heading or None,
                "excerpt": h.excerpt,
                "path": h.path,
            }
            for h in res.hits
        ],
        "linked": res.linked,
        "hint": "call get_note(slug) for the full note",
    }


@mcp.tool
def get_note(slug: str) -> dict:
    """Return the full markdown of a note by its slug or id."""
    note = get_note_by_slug(slug)
    if not note:
        return {"error": f"no note with slug/id '{slug}'"}
    return {
        "slug": note.slug,
        "frontmatter": note.meta.to_yaml_dict(),
        "body": note.body,
        "links": note.all_links(),
        "path": str(note.path) if note.path else None,
    }


@mcp.tool
def remember(
    title: str,
    body_md: str,
    type: str = "finding",
    tags: list[str] | None = None,
    links: list[str] | None = None,
    source: str | None = None,
    url: str | None = None,
    cwe: list[str] | None = None,
    program: str | None = None,
) -> dict:
    """Save a new note to the vault and index it immediately.

    Args:
        title: short note title.
        body_md: the note content as markdown. Use `[[slug]]` to link related notes.
        type: report | cve | technique | target | finding | writeup (default finding).
        tags: freeform tags.
        links: slugs of related notes to link.
        source / url: provenance.
        cwe: list like ["CWE-79"].
        program: bug bounty program / vendor.
    """
    if type not in NOTE_TYPES:
        return {"error": f"type must be one of {NOTE_TYPES}"}
    s = get_settings()
    from slugify import slugify

    base = slugify(title, max_length=70) or "note"
    now = datetime.now(UTC)
    note_id = f"{type[:4]}-{base}-{now:%Y%m%d%H%M%S}{now.microsecond // 1000:03d}"
    meta = Frontmatter(
        id=note_id,
        type=type,  # type: ignore[arg-type]
        title=title,
        source=source or "sift-remember",
        url=url,
        tags=tags or [],
        links=links or [],
        cwe=cwe or [],
        program=program,
        ingested=datetime.now(UTC),
    )
    note = Note(meta=meta, body=body_md)
    path = save_note(s.resolved_vault(), note, stamp=False)
    chunks = index_note(note)
    return {"saved": True, "slug": note.slug, "path": str(path), "chunks_indexed": chunks}


@mcp.tool
def list_notes(type: str | None = None, program: str | None = None, limit: int = 50) -> dict:
    """List notes (metadata only), newest first."""
    s = get_settings()
    rows = []
    for note in iter_notes(s.resolved_vault(), note_type=type if type in NOTE_TYPES else None):
        if program and (note.meta.program or "").lower() != program.lower():
            continue
        rows.append(
            {
                "slug": note.slug,
                "title": note.meta.title,
                "type": note.meta.type,
                "program": note.meta.program,
                "severity": note.meta.severity,
                "tags": note.meta.tags,
                "created": note.meta.created.isoformat() if note.meta.created else None,
            }
        )
    rows.sort(key=lambda r: r["created"] or "", reverse=True)
    return {"count": len(rows), "notes": rows[:limit]}


@mcp.tool
def stats() -> dict:
    """Vault + index statistics and last-ingest timestamps."""
    from sift.index.store import Store
    from sift.ingest.base import load_state

    s = get_settings()
    counts: dict[str, int] = {}
    for note in iter_notes(s.resolved_vault()):
        counts[note.meta.type] = counts.get(note.meta.type, 0) + 1
    return {
        "notes_by_type": counts,
        "total_notes": sum(counts.values()),
        "index_chunks": Store().count(),
        "last_ingest": load_state(),
        "vault_path": str(s.resolved_vault()),
    }


if __name__ == "__main__":
    mcp.run()
