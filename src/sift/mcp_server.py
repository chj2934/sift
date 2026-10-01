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


IDEA_STATUSES = ("hypothesis", "worked", "failed", "partial")


@mcp.tool
def capture_idea(
    idea: str,
    reasoning: str,
    target: str | None = None,
    tags: list[str] | None = None,
    cwe: list[str] | None = None,
    links: list[str] | None = None,
) -> dict:
    """Record a novel testing idea the moment it forms, before testing it.

    Call this during bug bounty work whenever you form a hypothesis worth trying -
    especially a specialized or non-obvious one. Ideas start as `hypothesis`; call
    `resolve_idea` afterwards to record what actually happened.

    Args:
        idea: the hypothesis in one or two sentences - what to try, and where.
        reasoning: why this might work here. What observation prompted it.
        target: program / host / component the idea is about.
        tags: freeform tags.
        cwe: list like ["CWE-79"].
        links: slugs of related notes (techniques, prior findings).
    """
    from slugify import slugify

    s = get_settings()
    now = datetime.now(UTC)
    base = slugify(idea, max_length=60) or "idea"
    meta = Frontmatter(
        id=f"idea-{base}-{now:%Y%m%d%H%M%S}{now.microsecond // 1000:03d}",
        type="technique",
        title=idea if len(idea) <= 120 else idea[:117] + "...",
        source="sift-capture-idea",
        program=target,
        tags=sorted({"idea", "status/hypothesis", *(tags or [])}),
        cwe=cwe or [],
        links=links or [],
        ingested=now,
        # extra is the filterable source of truth; the status/ tag mirrors it so
        # plain text search finds it too.
        extra={"status": "hypothesis", "target": target or "", "captured": now.isoformat()},
    )
    body = f"**Status:** hypothesis\n\n**Idea:** {idea}\n\n**Why here:** {reasoning}"
    note = Note(meta=meta, body=body)
    path = save_note(s.resolved_vault(), note, stamp=False)
    chunks = index_note(note)
    return {
        "saved": True,
        "slug": note.slug,
        "status": "hypothesis",
        "path": str(path),
        "chunks_indexed": chunks,
        "hint": "Call resolve_idea with this slug once you know whether it worked.",
    }


@mcp.tool
def resolve_idea(slug: str, status: str, notes: str) -> dict:
    """Record the outcome of a previously captured idea.

    Recording `failed` matters as much as `worked` - a documented dead end
    ("tried JWT alg confusion, RS256 validated properly") stops the next session
    re-testing it.

    Args:
        slug: the slug returned by capture_idea.
        status: worked | failed | partial.
        notes: what actually happened, and any detail worth keeping.
    """
    if status not in IDEA_STATUSES or status == "hypothesis":
        return {"error": f"status must be one of {IDEA_STATUSES[1:]}"}

    s = get_settings()
    note = get_note_by_slug(slug)
    if note is None:
        return {"error": f"no note with slug {slug!r}"}

    now = datetime.now(UTC)
    prior = str(note.meta.extra.get("status", "hypothesis"))
    note.meta.extra["status"] = status
    note.meta.extra["resolved"] = now.isoformat()
    note.meta.tags = sorted(
        {t for t in note.meta.tags if not t.startswith("status/")} | {f"status/{status}"}
    )
    note.meta.ingested = now
    note.body = (
        note.body.replace(f"**Status:** {prior}", f"**Status:** {status}", 1)
        + f"\n\n**Outcome ({status}):** {notes}"
    )

    from sift.index.store import Store

    save_note(s.resolved_vault(), note, stamp=False)
    store = Store()
    chunks = index_note(note, store)
    store.ensure_fts()
    return {"updated": True, "slug": note.slug, "status": status, "chunks_indexed": chunks}


@mcp.tool
def list_notes(
    type: str | None = None,
    program: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict:
    """List notes (metadata only), newest first.

    Args:
        type: report | cve | technique | target | finding | writeup.
        program: bug bounty program / vendor.
        status: for captured ideas - hypothesis | worked | failed | partial.
            Use `status="hypothesis"` to find ideas you never followed up on.
        limit: max rows returned.
    """
    s = get_settings()
    rows = []
    for note in iter_notes(s.resolved_vault(), note_type=type if type in NOTE_TYPES else None):
        if program and (note.meta.program or "").lower() != program.lower():
            continue
        if status and str(note.meta.extra.get("status", "")).lower() != status.lower():
            continue
        rows.append(
            {
                "slug": note.slug,
                "title": note.meta.title,
                "type": note.meta.type,
                "program": note.meta.program,
                "severity": note.meta.severity,
                "status": note.meta.extra.get("status"),
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
