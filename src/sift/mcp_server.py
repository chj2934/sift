"""MCP server exposing the vault memory as tools for Claude Code / Claude Desktop.

Run with ``sift mcp`` or ``python -m sift.mcp_server`` (stdio transport); both go
through `main`. The server holds no LLM - Claude (the host) does the reasoning; this
retrieves, reads and persists notes.

How it behaves
--------------
* **stdout is the JSON-RPC wire.** Nothing here prints. Diagnostics go to the
  ``sift`` logger, which `main` points at stderr.
* **Cheap startup.** `sift.pipeline` (and with it the index, LanceDB and the
  embedder) is imported inside the tools, so ``initialize`` is answered quickly. Once
  the client has initialised, `main`'s server warms the vault catalog, the embedder
  and the link graph on daemon threads. Tests - which call the tools directly or
  through an in-memory client - never start those threads.
* **No vault scans per call.** Lookups and listings read the vault catalog
  (`sift.vault.catalog`), refreshed by a stat walk. A note reference resolves
  strongest-first: exact id > slug > legacy 80-char slug > filename, or a vault path
  (only a catalogued note). An ambiguous reference is reported to a reader with its
  candidates and refused by a writer, which never edits a guessed note.
* **Writes are upserts under one lock.** Every write holds the vault write lock
  (process and cross-process) for its whole read-modify-write-index, so concurrent
  calls cannot lose an update. Saves go through `notes.write_note`, an upsert by id:
  a note keeps the file it lives in, even one the user renamed in Obsidian.
  `remember` appends to a note it already wrote under the same title instead of
  forking ``Title (2).md``. New ids are at most 80 characters, so an id is its own
  slug and never collides with another note's truncated one.
* **Errors raise `ToolError`** (``isError`` on the wire). An ambiguous read is not an
  error: it returns the candidates.
* MCP tools never compact or rebuild the index (`Store.optimize` is CLI-only); new
  rows are keyword-searchable without an FTS rebuild.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import threading
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal
from urllib.parse import urlsplit

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from pydantic import Field

from sift.config import get_settings
from sift.vault.schema import NOTE_TYPES, Frontmatter, NoteType

if TYPE_CHECKING:
    from sift.pipeline import NoteLookup
    from sift.vault.catalog import CatalogRow, VaultCatalog
    from sift.vault.notes import Note, SaveResult

log = logging.getLogger(__name__)

IDEA_STATUSES = ("hypothesis", "worked", "failed", "partial")
IdeaStatus = Literal["hypothesis", "worked", "failed", "partial"]
IdeaOutcome = Literal["worked", "failed", "partial"]
ListSort = Literal["created", "recent"]

# A generated id is at most this long (the legacy slug cap), so it is its own slug in
# both the old and the new slug scheme.
_ID_MAX = 80
_TS_DIGITS = 17  # %Y%m%d%H%M%S + milliseconds
# Ids minted by `remember`: `<type[:4]>-<title slug>-<17-digit timestamp>`.
_REMEMBER_ID = re.compile(r"^[a-z]{3,4}-.+-\d{17}$")

# MCP tool annotations (hints to the client; FastMCP passes them through).
_READ = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
_WRITE = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}
_REPLACE = {
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": False,
    "openWorldHint": False,
}
_FORGET = {
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": True,
    "openWorldHint": False,
}
_FETCH = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
}

_SKIPPED_PATHS_SHOWN = 20


# --------------------------------------------------------------------------------
# Background work (only in a served process)
# --------------------------------------------------------------------------------

_background_lock = threading.Lock()
_background_enabled = False
_background_started = False

_FALSE = frozenset({"0", "false", "no", "off", ""})


def _flag(field: str, env: str, default: bool) -> bool:
    """A boolean setting. Read from settings when config.py declares it, else from
    the environment directly, so the switch works before the setting is declared."""
    value = getattr(get_settings(), field, None)
    if value is None:
        raw = os.environ.get(env)
        if raw is None or not raw.strip():
            return default
        value = raw
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE
    return bool(value)


def _warmup_on() -> bool:
    return _flag("mcp_warmup", "SIFT_MCP_WARMUP", True)


def _auto_sync_on() -> bool:
    return _flag("mcp_auto_sync", "SIFT_MCP_AUTO_SYNC", True)


def _warm_vault() -> None:
    """Build (or load and re-validate) the vault catalog, then the link graph."""
    try:
        vault = get_settings().resolved_vault()
        if not vault.is_dir():
            return
        from sift.vault.catalog import get_catalog

        get_catalog(vault).ensure_fresh()
        if _warmup_on():
            from sift.index.graph import build_link_index

            build_link_index(vault)
    except Exception as exc:  # noqa: BLE001 - a warm-up must never take the server down
        log.warning("vault warm-up failed (%s: %s)", type(exc).__name__, exc)


def _warm_models() -> None:
    """Import the index stack and load the query model, then start the index sync."""
    try:
        from sift import pipeline

        if _warmup_on():
            from sift.index import embed, rerank

            embed.warmup(query=True)
            rerank.warmup()
        if _auto_sync_on():
            pipeline.sync_in_background(min_interval=0)
    except Exception as exc:  # noqa: BLE001
        log.warning("model warm-up failed (%s: %s)", type(exc).__name__, exc)


def enable_background() -> None:
    """Allow `start_background` (`main` calls this before serving)."""
    global _background_enabled
    with _background_lock:
        _background_enabled = True


def start_background() -> list[threading.Thread]:
    """Warm caches and models off the request path, once per process.

    A no-op unless `main` enabled it, so tools driven by tests never load a model.
    The model load is single-flight (`embed.warmup`): a tool call that arrives
    mid-warm-up waits for it instead of loading a second copy. With
    ``SIFT_MCP_WARMUP=false`` only the vault catalog is warmed (no VRAM).
    """
    global _background_started
    with _background_lock:
        if not _background_enabled or _background_started:
            return []
        _background_started = True
    threads = [
        threading.Thread(target=_warm_vault, name="sift-warm-vault", daemon=True),
        threading.Thread(target=_warm_models, name="sift-warm-models", daemon=True),
    ]
    for t in threads:
        t.start()
    return threads


def _maybe_sync() -> None:
    """Let a vault edit made in Obsidian reach search: a debounced, single-flight
    background sync (a no-op stat walk when nothing changed). Served process only."""
    if not _background_enabled or not _auto_sync_on():
        return
    try:
        from sift.pipeline import sync_in_background

        sync_in_background()
    except Exception as exc:  # noqa: BLE001
        log.warning("index sync not started (%s)", exc)


class _StartWhenServing(Middleware):
    """Start the warm-up once the client has initialised: the handshake is answered
    first, and by then stdio has diverted fd 1 away from the wire."""

    async def on_initialize(self, context: Any, call_next: Any) -> Any:
        result = await call_next(context)
        start_background()
        return result

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        start_background()  # a client that skipped initialize; normally a no-op
        return await call_next(context)


mcp = FastMCP(
    name="sift",
    instructions=(
        "Personal bug-bounty memory: disclosed reports, CVEs, techniques, targets, and the "
        "user's own findings, stored as an Obsidian-style markdown vault. Use `search_memory` "
        "before answering questions about vulnerabilities, past reports, or techniques. Use "
        "`remember` to save durable knowledge (a new technique, a finding, notes on a target); "
        "it appends to a note it already wrote under the same title. Notes are addressed by "
        "`note_id` (returned by every tool). Only operate against targets the user is "
        "explicitly authorized to test."
    ),
    middleware=[_StartWhenServing()],
)


# --------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------


def _timestamp(now: datetime) -> str:
    return f"{now:%Y%m%d%H%M%S}{now.microsecond // 1000:03d}"


def _new_note_id(prefix: str, text: str, fallback: str, now: datetime) -> str:
    """``<prefix>-<slug of text>-<17-digit timestamp>``, at most 80 characters.

    The slug is cut to fit, so the id is its own slug (`Note.slug == id`) and the
    timestamp survives the legacy 80-character slug cap that once made two notes
    share one slug.
    """
    from slugify import slugify

    budget = _ID_MAX - len(prefix) - _TS_DIGITS - 2
    base = slugify(text or "", max_length=budget) or fallback
    return f"{prefix}-{base}-{_timestamp(now)}"


def _check_range(name: str, value: int, lo: int, hi: int) -> None:
    if not lo <= int(value) <= hi:
        raise ToolError(f"{name} must be between {lo} and {hi}, got {value}")


def _check_type(value: str | None, name: str = "type") -> None:
    if value is not None and value not in NOTE_TYPES:
        raise ToolError(f"{name} must be one of {', '.join(NOTE_TYPES)}; got {value!r}")


def _parse_day(value: str | None, name: str) -> date | None:
    if value is None or not str(value).strip():
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        raise ToolError(f"{name} must be an ISO date (YYYY-MM-DD); got {value!r}") from None


def _union(old: Iterable[str], new: Iterable[str] | None) -> list[str]:
    """Order-preserving union of two string lists (exact matches only)."""
    seen: dict[str, None] = {}
    for item in [*old, *(new or [])]:
        s = str(item).strip()
        if s:
            seen.setdefault(s, None)
    return list(seen)


def _without(items: Iterable[str], drop: Iterable[str] | None) -> list[str]:
    gone = {str(d).strip().casefold() for d in drop or ()}
    return [i for i in items if str(i).strip().casefold() not in gone]


def _same_text(a: str | None, b: str | None) -> bool:
    return (a or "").strip().casefold() == (b or "").strip().casefold()


def _day_of_ns(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, UTC).date().isoformat()


def _iso_of_ns(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, UTC).isoformat(timespec="seconds")


def _effective_day(row: CatalogRow) -> str:
    """The note's date for "newest first": its `created` date, else the day it was
    last written. remember/capture_idea notes written before `created` was stamped
    have none, and used to sort after every dated note."""
    return row.created or _day_of_ns(row.mtime_ns)


def _sort_key(row: CatalogRow, sort: str) -> tuple:
    if sort == "recent":
        return (row.mtime_ns,)
    return (_effective_day(row), row.mtime_ns)


def _created_of_ts(ts: float) -> str | None:
    if not ts or ts <= 0:
        return None
    try:
        return datetime.fromtimestamp(ts, UTC).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _append_update(body: str, text: str, now: datetime) -> str:
    return f"{body.rstrip()}\n\n## Update {now:%Y-%m-%d %H:%M} UTC\n\n{text.strip()}\n"


_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t#]*$")


def _headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line index, level, text) of each markdown heading outside fenced code."""
    out: list[tuple[int, int, str]] = []
    fence: str | None = None
    for i, line in enumerate(lines):
        m = _FENCE.match(line)
        if m:
            mark = m.group(1)
            if fence is None:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        h = _HEADING.match(line)
        if h and h.group(2).strip():
            out.append((i, len(h.group(1)), h.group(2).strip()))
    return out


def _extract_section(body: str, wanted: str) -> str | None:
    """The section under the heading matching `wanted` (exact, else substring,
    case-insensitive), up to the next heading of the same or a higher level."""
    want = wanted.strip().lstrip("#").strip().casefold()
    if not want:
        return None
    lines = body.splitlines()
    hs = _headings(lines)
    pick = next((h for h in hs if h[2].casefold() == want), None) or next(
        (h for h in hs if want in h[2].casefold()), None
    )
    if pick is None:
        return None
    start, level, _text = pick
    end = next((i for i, lv, _t in hs if i > start and lv <= level), len(lines))
    return "\n".join(lines[start:end]).strip()


def _one_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return (text[0] if text else type(exc).__name__)[:300]


_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")


def _local_host(host: str | None) -> str | None:
    """Why `host` is a local target, judged from the name alone (no DNS): loopback,
    private, link-local (cloud metadata) and other non-public IP literals, including
    the all-digits form, and local-only names. The fetcher does the full check, with
    resolution and every redirect hop; this stops the obvious cases before any I/O."""
    import ipaddress

    h = (host or "").strip().strip("[]").rstrip(".").lower()
    if not h:
        return "no host"
    if h == "localhost" or h.endswith(_LOCAL_SUFFIXES):
        return "a local host name"
    try:
        ip = ipaddress.ip_address(int(h)) if h.isdigit() else ipaddress.ip_address(h)
    except ValueError:
        return None
    if not ip.is_global or ip.is_multicast:
        return "a private, loopback or link-local address"
    return None


# --------------------------------------------------------------------------------
# Vault access
# --------------------------------------------------------------------------------


def _vault() -> Path:
    return get_settings().resolved_vault()


@contextlib.contextmanager
def _vault_lock(vault: Path) -> Iterator[None]:
    """The vault write lock, turned into a tool error when another sift process (a
    CLI ingest) holds it past the timeout. Errors raised inside pass through."""
    from sift.vault.notes import write_lock

    stack = contextlib.ExitStack()
    try:
        stack.enter_context(write_lock(vault))
    except RuntimeError as exc:
        raise ToolError(f"{exc}. Try again once it finishes.") from exc
    with stack:
        yield


def _catalog(vault: Path, *, max_age: float = 1.0) -> VaultCatalog:
    from sift.vault.catalog import fresh_catalog

    return fresh_catalog(vault, max_age=max_age)


def _lookup(key: str, *, max_age: float) -> NoteLookup:
    from sift.pipeline import resolve_note

    return resolve_note(key, vault=_vault(), max_age=max_age)


def _ambiguous_message(key: str, lookup: NoteLookup) -> str:
    ids = ", ".join(c["note_id"] for c in lookup.candidate_info()[:10])
    return (
        f"{key!r} is ambiguous: {len(lookup.candidates)} notes answer to it ({ids}). "
        "Nothing was written; pass the exact note_id."
    )


def _resolve_for_write(key: str) -> Note:
    """The one note `key` names, fresh from disk. Called under the vault lock. A
    writer never guesses: an ambiguous or unknown reference is an error."""
    if not str(key or "").strip():
        raise ToolError("a note_id is required")
    lookup = _lookup(key, max_age=0.0)
    if lookup.ambiguous:
        raise ToolError(_ambiguous_message(key, lookup))
    if lookup.note is None:
        raise ToolError(f"no note with id or slug {key!r}")
    return lookup.note


def _write(vault: Path, note: Note, *, stamp: bool = False) -> SaveResult:
    from sift.vault.notes import IdConflict, write_note

    try:
        return write_note(vault, note, stamp=stamp)
    except IdConflict as exc:  # a fresh id cannot clash; never overwrite if it does
        raise ToolError(str(exc)) from exc


def _is_user_note(note: Note) -> bool:
    from sift.quality import is_user_authored

    return is_user_authored(note.meta)


def _is_idea(note: Note) -> bool:
    m = note.meta
    return "idea" in m.tags or m.source == "sift-capture-idea" or m.id.startswith("idea-")


def _unused_id(vault: Path, prefix: str, text: str, fallback: str, now: datetime) -> str:
    """A generated id no file carries yet. Called under the vault lock: two calls in
    one millisecond would otherwise mint one id, and the second save - an upsert by
    id - would overwrite the first note."""
    from sift.vault.notes import locate_note

    for _ in range(1000):
        note_id = _new_note_id(prefix, text, fallback, now)
        if locate_note(vault, note_id) is None:
            return note_id
        now += timedelta(milliseconds=1)
    raise ToolError("could not mint an unused note id")


def _remembered_twin(vault: Path, type_: str, title: str, program: str | None) -> Note | None:
    """The note `remember` already wrote with this type, title and program (the most
    recently written one), loaded fresh; None if there is none.

    Matches notes remember authored (its id pattern or source) only - never an
    ingested note, a captured idea or a note of another program - so a generic title
    reused on another program still gets its own note.
    """
    prefix = f"{type_[:4]}-"
    rows = [
        r
        for r in _catalog(vault, max_age=0.0).rows()
        if r.type == type_
        and _same_text(r.title, title)
        and _same_text(r.program, program)
        and r.source != "sift-capture-idea"
        and r.id.startswith(prefix)
        and (r.source == "sift-remember" or _REMEMBER_ID.match(r.id))
    ]
    from sift.vault.notes import load_note

    # Newest first. Load the exact catalogued file (fresh walk, under the lock): an
    # id lookup would call an id that two old files share ambiguous, and fork again.
    for row in sorted(rows, key=lambda r: r.mtime_ns, reverse=True):
        try:
            note = load_note(row.path)
        except Exception:  # noqa: BLE001 - vanished or unreadable: try the next one
            continue
        if note.meta.id == row.id:
            return note
    return None


def _note_payload(note: Note, res: SaveResult | None = None) -> dict:
    out: dict[str, Any] = {
        "note_id": note.meta.id,
        "slug": note.slug,
        "title": note.meta.title,
        "type": note.meta.type,
        "path": str(note.path) if note.path else None,
    }
    if res is not None and res.renamed and res.previous_path is not None:
        out["renamed_from"] = str(res.previous_path)
    return out


_URL_INDEX: dict[str, tuple[int, dict[str, CatalogRow]]] = {}
_URL_INDEX_LOCK = threading.Lock()


def _note_with_url(vault: Path, url: str) -> CatalogRow | None:
    """The first note (path order) whose canonical url is `url`'s, from the catalog;
    the url map is rebuilt only when the catalog changed."""
    from sift.vault.notes import canonical_url

    want = canonical_url(url)
    if not want:
        return None
    cat = _catalog(vault, max_age=0.0)
    gen = cat.generation
    key = str(vault)
    with _URL_INDEX_LOCK:
        cached = _URL_INDEX.get(key)
        if cached is None or cached[0] != gen:
            by_url: dict[str, CatalogRow] = {}
            for row in cat.rows():
                if row.url:
                    by_url.setdefault(canonical_url(row.url), row)
            cached = _URL_INDEX[key] = (gen, by_url)
    return cached[1].get(want)


# --------------------------------------------------------------------------------
# Read tools
# --------------------------------------------------------------------------------


@mcp.tool(annotations=_READ)
def search_memory(
    query: str = "",
    queries: list[str] | None = None,
    k: Annotated[int, Field(ge=1, le=50)] = 8,
    type: NoteType | None = None,
    cwe: str | None = None,
    program: str | None = None,
    min_quality: Annotated[int, Field(ge=0, le=100)] = 0,
    expand_links: bool = False,
) -> dict:
    """Hybrid (semantic + keyword) search over the memory vault.

    Prefer `queries` with 2-4 phrasings of the same need, searched together and fused
    into one ranking: (1) the exact identifiers - function, endpoint, parameter, header,
    CVE id, error string - which the keyword side matches literally; (2) a plain-language
    description; optionally (3) a sentence written as the note you hope exists. Each
    hit then lists `matched_queries`: a note found by several phrasings is the strongest
    match. Use `query` alone for a single exact lookup.

    Args:
        query: one natural-language or keyword query.
        queries: several phrasings (up to 6), fused into one ranking. Combined with
            `query` if both are given; blanks and repeats are dropped.
        k: number of notes to return (1-50, default 8).
        type: optional note type filter.
        cwe: optional CWE filter, e.g. "CWE-79" (exact match).
        program: optional bug bounty program / vendor filter (case-insensitive).
        min_quality: drop hits below this 0-100 heuristic quality score (default 0 = off).
        expand_links: also return notes linked (1 hop) from the top hits.
    """
    from sift.index.store import MAX_QUERIES, normalize_queries

    phrasings = normalize_queries(query, queries)
    if not phrasings:
        raise ToolError("query is empty: pass `query` or a non-empty `queries` list")
    if len(phrasings) > MAX_QUERIES:
        raise ToolError(
            f"at most {MAX_QUERIES} queries per call (got {len(phrasings)}); "
            "keep the 2-4 phrasings that differ most"
        )
    _check_range("k", k, 1, 50)
    _check_range("min_quality", min_quality, 0, 100)
    _check_type(type)
    _maybe_sync()

    from sift.pipeline import search

    multi = len(phrasings) > 1
    filters = {k2: v for k2, v in {"type": type, "cwe": cwe, "program": program}.items() if v}
    try:
        res = search(
            phrasings if multi else phrasings[0],
            k=k,
            filters=filters or None,
            expand_links=expand_links,
            min_quality=min_quality,
        )
    except (ValueError, RuntimeError) as exc:  # bad filter, model/index mismatch, failed search
        raise ToolError(str(exc)) from exc

    def _row(h) -> dict:
        row = {
            "note_id": h.note_id,
            "slug": h.slug,
            "title": h.title,
            "type": h.type,
            "severity": h.severity or None,
            "program": h.program or None,
            "url": h.url or None,
            "created": _created_of_ts(h.created_ts),
            "quality": h.quality,
            "score": round(h.score, 4),
            "matched_section": h.heading or None,
            "excerpt": h.excerpt,
            "path": h.path,
        }
        if multi:
            row["matched_queries"] = list(h.matched_queries)
        return row

    out: dict = {"queries": phrasings} if multi else {"query": phrasings[0]}
    out.update(
        {
            "results": [_row(h) for h in res.hits],
            "linked": res.linked,
            "warnings": list(res.warnings),
            "hint": "call get_note(note_id) for the full note",
        }
    )
    return out


@mcp.tool(annotations=_READ)
def get_note(
    slug: str,
    section: str | None = None,
    max_chars: Annotated[int, Field(ge=200)] | None = None,
) -> dict:
    """Return a note's full markdown.

    Args:
        slug: the note_id (preferred), slug, filename, or the vault path that
            search_memory returned. If several notes answer to it, the candidates are
            returned instead of a guess.
        section: return only the section under this heading (case-insensitive).
        max_chars: cut the body to this many characters (at least 200).
    """
    if max_chars is not None and max_chars < 200:
        raise ToolError("max_chars must be at least 200")
    lookup = _lookup(slug, max_age=1.0)
    if lookup.ambiguous:
        return {
            "error": "ambiguous slug",
            "candidates": lookup.candidate_info(),
            "hint": "call get_note with one of the candidates' note_id",
        }
    note = lookup.note
    if note is None:
        raise ToolError(f"no note with id or slug {slug!r}")
    body = note.body
    out: dict[str, Any] = {
        "note_id": note.meta.id,
        "slug": note.slug,
        "frontmatter": note.meta.to_yaml_dict(),
        "links": note.all_links(),
        "path": str(note.path) if note.path else None,
    }
    if section:
        part = _extract_section(body, section)
        if part is None:
            heads = [t for _i, _lv, t in _headings(body.splitlines())]
            raise ToolError(
                f"no heading in {note.meta.id!r} matches {section!r}; headings: "
                + (", ".join(heads[:30]) or "(none)")
            )
        out["section"] = section
        body = part
    if max_chars is not None and len(body) > max_chars:
        out["truncated"] = True
        out["body_chars"] = len(body)
        body = body[:max_chars]
    out["body"] = body
    return out


@mcp.tool(annotations=_READ)
def list_notes(
    type: NoteType | None = None,
    program: str | None = None,
    status: IdeaStatus | None = None,
    tag: str | None = None,
    source: str | None = None,
    since: str | None = None,
    sort: ListSort = "created",
    limit: Annotated[int, Field(ge=1, le=500)] = 50,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> dict:
    """List notes (metadata only), newest first.

    Args:
        type: note type filter.
        program: bug bounty program / vendor (case-insensitive).
        status: for captured ideas - hypothesis | worked | failed | partial.
            Use `status="hypothesis"` to find ideas you never followed up on.
        tag: only notes carrying this tag (case-insensitive).
        source: only notes from this source, e.g. "sift-remember" or
            "sift-capture-idea" for what you recorded yourself.
        since: only notes dated on or after this ISO date.
        sort: "created" (default) - newest by created date, falling back to when the
            note was last written; "recent" - most recently written first.
        limit: max rows returned (1-500).
        offset: rows to skip, for paging.
    """
    _check_type(type)
    if status is not None and status not in IDEA_STATUSES:
        raise ToolError(f"status must be one of {', '.join(IDEA_STATUSES)}")
    if sort not in ("created", "recent"):
        raise ToolError("sort must be 'created' or 'recent'")
    _check_range("limit", limit, 1, 500)
    if offset < 0:
        raise ToolError("offset must be >= 0")
    since_day = _parse_day(since, "since")

    picked: list[CatalogRow] = []
    for row in _catalog(_vault()).rows():
        if type and row.type != type:
            continue
        if program and not _same_text(row.program, program):
            continue
        if status and not _same_text(row.status, status):
            continue
        if tag and not any(_same_text(t, tag) for t in row.tags):
            continue
        if source and not _same_text(row.source, source):
            continue
        if since_day:
            day = _day_of_ns(row.mtime_ns) if sort == "recent" else _effective_day(row)
            if day < since_day.isoformat():
                continue
        picked.append(row)
    picked.sort(key=lambda r: _sort_key(r, sort), reverse=True)
    page = picked[offset : offset + limit]
    return {
        "count": len(picked),
        "offset": offset,
        "notes": [
            {
                "note_id": r.id,
                "slug": r.slug,
                "title": r.title,
                "type": r.type,
                "program": r.program,
                "severity": r.severity,
                "status": r.status,
                "tags": list(r.tags),
                "created": r.created,
                "updated": _iso_of_ns(r.mtime_ns),
                "path": r.rel,
            }
            for r in page
        ],
    }


@mcp.tool(annotations=_READ)
def stats() -> dict:
    """Vault and index statistics, plus the last ingest of each source."""
    from sift.index.store import Store
    from sift.ingest.base import load_state

    s = get_settings()
    vault = s.resolved_vault()
    cat = _catalog(vault)
    counts = cat.counts_by_type()
    skipped = cat.skipped()
    out: dict[str, Any] = {
        "notes_by_type": counts,
        "total_notes": sum(counts.values()),
        "index_chunks": Store().count(),
        "last_ingest": load_state(),
        "vault_path": str(vault),
        # Files that look like notes but cannot be used (empty, no frontmatter,
        # invalid). Paths only: the reasons can quote file content.
        "skipped_notes": len(skipped),
        "skipped_paths": [_rel(p, vault) for p, _why in skipped[:_SKIPPED_PATHS_SHOWN]],
        "duplicate_id_groups": len(cat.duplicate_ids()),
    }
    if out["index_chunks"]:
        try:
            from sift.pipeline import stale_index_reason

            stale = stale_index_reason(model=s.embed_model)
        except Exception as exc:  # noqa: BLE001 - informational only
            log.debug("stale-index check failed: %s", exc)
            stale = None
        if stale:
            out["index_warning"] = f"{stale}; run `sift reindex --force` once"
    return out


def _rel(path: Path, vault: Path) -> str:
    try:
        return Path(path).relative_to(vault).as_posix()
    except ValueError:
        return Path(path).name


# --------------------------------------------------------------------------------
# Write tools
# --------------------------------------------------------------------------------


@mcp.tool(annotations=_WRITE)
def remember(
    title: str,
    body_md: str,
    type: NoteType = "finding",
    tags: list[str] | None = None,
    links: list[str] | None = None,
    source: str | None = None,
    url: str | None = None,
    cwe: list[str] | None = None,
    program: str | None = None,
    severity: str | None = None,
    created: str | None = None,
    note_id: str | None = None,
) -> dict:
    """Save durable knowledge to the vault and index it immediately.

    Never forks a note. With `note_id`, `body_md` is appended to that note as a dated
    "Update" section. Without it, if you already remembered a note of this type with
    this exact title (and program), the text is appended there (`merged_into`).
    Otherwise a new note is written. Tags, links and CWEs are merged. To rewrite a
    body, retitle or untag a note use `update_note`; to drop one use `forget_note`.

    Args:
        title: short note title (specific: it is also how a later call finds it).
        body_md: the note content as markdown. Use `[[note_id]]` to link notes.
        type: note type (default finding).
        tags: freeform tags.
        links: ids or slugs of related notes.
        source / url: provenance.
        cwe: list like ["CWE-79"].
        program: bug bounty program / vendor.
        severity: critical | high | medium | low | none | info.
        created: ISO date of the underlying thing (e.g. a report's disclosure);
            defaults to today for a new note.
        note_id: append to this existing note instead (your own notes only).
    """
    title = (title or "").strip()
    if not title:
        raise ToolError("title is required")
    if not (body_md or "").strip():
        raise ToolError("body_md is empty")
    _check_type(type)
    created_day = _parse_day(created, "created")

    from sift.pipeline import index_note
    from sift.quality import AUTHORED_VIA
    from sift.vault.notes import Note

    now = datetime.now(UTC)
    vault = _vault()
    with _vault_lock(vault):
        if note_id:
            target = _resolve_for_write(note_id)
            if not _is_user_note(target):
                raise ToolError(
                    f"{target.meta.id!r} is a {target.meta.source or 'ingested'} note; a re-ingest "
                    "would overwrite text appended to it. Remember a new note and link it "
                    f"(links=[{target.meta.id!r}]) instead."
                )
            merged = False
        else:
            target = _remembered_twin(vault, type, title, program)
            merged = target is not None
        if target is not None:
            target.body = _append_update(target.body, body_md, now)
            m = target.meta
            m.tags = _union(m.tags, tags)
            m.links = _union(m.links, links)
            changes: dict[str, Any] = {"cwe": _union(m.cwe, cwe)}
            if severity:
                changes["severity"] = severity
            if program and not m.program:
                changes["program"] = program
            if url and not m.url:
                changes["url"] = url
            if created_day:
                changes["created"] = created_day
            _set_fields(target, changes)
            m.ingested = now
            res = _write(vault, target)
            chunks = index_note(target)
            out = {"saved": True, "updated": True, **_note_payload(target, res)}
            out["chunks_indexed"] = chunks
            if merged:
                out["merged_into"] = target.meta.id
                out["hint"] = (
                    "Appended to the note you already remembered under this title. Use a "
                    "more specific title for a separate note, or update_note to rewrite it."
                )
            return out

        new_id = _unused_id(vault, type[:4], title, "note", now)
        meta = Frontmatter(
            id=new_id,
            type=type,  # type: ignore[arg-type]
            title=title,
            source=source or "sift-remember",
            url=url,
            tags=tags or [],
            links=links or [],
            cwe=cwe or [],
            program=program,
            severity=severity,
            created=created_day or now.date(),
            ingested=now,
            # The durable "the user wrote this" marker (prune keeps it, search floors
            # its quality), set even when the caller passes another `source`.
            extra={AUTHORED_VIA: "sift-remember"},
        )
        note = Note(meta=meta, body=body_md)
        res = _write(vault, note)
        chunks = index_note(note)
    return {"saved": True, "updated": False, **_note_payload(note, res), "chunks_indexed": chunks}


def _set_fields(note: Note, changes: dict[str, Any]) -> None:
    """Assign frontmatter fields through validation (severity and CWE normalising),
    keeping everything else on the loaded note - unknown keys, invalid values kept
    verbatim - untouched. List fields are de-duplicated after normalising, so
    ``352`` merged into ``CWE-352`` stays one entry."""
    if not changes:
        return
    m = note.meta
    probe = Frontmatter.model_validate(
        {"id": m.id, "type": m.type, "title": changes.get("title", m.title), **changes}
    )
    for key in changes:
        value = getattr(probe, key)
        if isinstance(value, list):
            value = list(dict.fromkeys(value))
        setattr(m, key, value)


@mcp.tool(annotations=_WRITE)
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
    `resolve_idea` with the returned note_id afterwards to record what happened.

    Args:
        idea: the hypothesis in one or two sentences - what to try, and where.
        reasoning: why this might work here. What observation prompted it.
        target: program / host / component the idea is about.
        tags: freeform tags.
        cwe: list like ["CWE-79"].
        links: ids or slugs of related notes (techniques, prior findings).
    """
    idea = (idea or "").strip()
    if not idea:
        raise ToolError("idea is required")

    from sift.pipeline import index_note
    from sift.quality import AUTHORED_VIA
    from sift.vault.notes import Note

    now = datetime.now(UTC)
    vault = _vault()
    with _vault_lock(vault):
        meta = Frontmatter(
            id=_unused_id(vault, "idea", idea, "idea", now),
            type="technique",
            title=idea if len(idea) <= 120 else idea[:117] + "...",
            source="sift-capture-idea",
            program=target,
            tags=sorted({"idea", "status/hypothesis", *(tags or [])}),
            cwe=cwe or [],
            links=links or [],
            created=now.date(),
            ingested=now,
            # extra is the filterable source of truth; the status/ tag mirrors it so
            # plain text search finds it too.
            extra={
                "status": "hypothesis",
                "target": target or "",
                "captured": now.isoformat(),
                AUTHORED_VIA: "sift-capture-idea",
            },
        )
        body = f"**Status:** hypothesis\n\n**Idea:** {idea}\n\n**Why here:** {reasoning}"
        note = Note(meta=meta, body=body)
        res = _write(vault, note)
        chunks = index_note(note)
    return {
        "saved": True,
        **_note_payload(note, res),
        "status": "hypothesis",
        "chunks_indexed": chunks,
        "hint": "Call resolve_idea with this note_id once you know whether it worked.",
    }


@mcp.tool(annotations=_WRITE)
def resolve_idea(slug: str, status: IdeaOutcome, notes: str) -> dict:
    """Record the outcome of a previously captured idea.

    Recording `failed` matters as much as `worked` - a documented dead end
    ("tried JWT alg confusion, RS256 validated properly") stops the next session
    re-testing it.

    Args:
        slug: the note_id (or slug) returned by capture_idea.
        status: worked | failed | partial.
        notes: what actually happened, and any detail worth keeping.
    """
    if status not in IDEA_STATUSES[1:]:
        raise ToolError(f"status must be one of {', '.join(IDEA_STATUSES[1:])}")

    from sift.pipeline import index_note

    vault = _vault()
    with _vault_lock(vault):
        note = _resolve_for_write(slug)
        if not _is_idea(note):
            raise ToolError(
                f"{note.meta.id!r} is a {note.meta.type} note, not a captured idea; nothing written"
            )
        now = datetime.now(UTC)
        extra = note.meta.extra if isinstance(note.meta.extra, dict) else {}
        prior = str(extra.get("status", "hypothesis"))
        extra["status"] = status
        extra["resolved"] = now.isoformat()
        note.meta.extra = extra
        note.meta.tags = sorted(
            {t for t in note.meta.tags if not t.startswith("status/")} | {f"status/{status}"}
        )
        note.meta.ingested = now
        note.body = (
            note.body.replace(f"**Status:** {prior}", f"**Status:** {status}", 1)
            + f"\n\n**Outcome ({status}):** {notes}"
        )
        # Back into the file it was loaded from (even one renamed in Obsidian); new
        # rows are keyword-searchable without an FTS rebuild.
        res = _write(vault, note)
        chunks = index_note(note)
    return {"updated": True, **_note_payload(note, res), "status": status, "chunks_indexed": chunks}


@mcp.tool(annotations=_REPLACE)
def update_note(
    note_id: str,
    body_md: str | None = None,
    append_md: str | None = None,
    title: str | None = None,
    tags_add: list[str] | None = None,
    tags_remove: list[str] | None = None,
    links_add: list[str] | None = None,
    cwe_add: list[str] | None = None,
    program: str | None = None,
    severity: str | None = None,
    url: str | None = None,
    force: bool = False,
) -> dict:
    """Correct a note in place: same id, same file, index rows replaced.

    For notes you wrote (remember, capture_idea, the user's own). An ingested note
    (a CVE, a disclosed report) needs `force=True`, and a later re-ingest of its
    source may overwrite the change - prefer remembering a linked note.

    Args:
        note_id: the note to change (exact id preferred).
        body_md: replace the whole body.
        append_md: append a dated "Update" section.
        title: new title (the file is renamed only if sift named it).
        tags_add / tags_remove: tags to add or remove.
        links_add: ids or slugs of related notes to link.
        cwe_add: CWEs to add, like ["CWE-79"].
        program / severity / url: set these fields.
        force: allow changing a note that is not user-authored.
    """
    edits = {
        "body_md": body_md,
        "append_md": append_md,
        "title": title,
        "tags_add": tags_add,
        "tags_remove": tags_remove,
        "links_add": links_add,
        "cwe_add": cwe_add,
        "program": program,
        "severity": severity,
        "url": url,
    }
    if all(v is None for v in edits.values()):
        raise ToolError("nothing to change")
    if title is not None and not title.strip():
        raise ToolError("title cannot be empty")

    from sift.pipeline import index_note

    now = datetime.now(UTC)
    vault = _vault()
    with _vault_lock(vault):
        note = _resolve_for_write(note_id)
        if not force and not _is_user_note(note):
            raise ToolError(
                f"{note.meta.id!r} is not a note you wrote (source {note.meta.source or '-'}); "
                "pass force=True to change it anyway (a re-ingest may overwrite the change), "
                "or remember a linked note instead"
            )
        m = note.meta
        changed: list[str] = []
        if body_md is not None:
            note.body = body_md
            changed.append("body")
        if append_md is not None and append_md.strip():
            note.body = _append_update(note.body, append_md, now)
            changed.append("append")
        fields: dict[str, Any] = {}
        if title is not None:
            fields["title"] = title.strip()
        if tags_add or tags_remove:
            fields["tags"] = _without(_union(m.tags, tags_add), tags_remove)
        if links_add:
            fields["links"] = _union(m.links, links_add)
        if cwe_add:
            fields["cwe"] = _union(m.cwe, cwe_add)
        for key, value in (("program", program), ("severity", severity), ("url", url)):
            if value is not None:
                fields[key] = value.strip() or None
        try:
            _set_fields(note, fields)
        except ValueError as exc:  # pydantic ValidationError
            raise ToolError(f"invalid value: {_one_line(exc)}") from exc
        changed += list(fields)
        m.ingested = now
        res = _write(vault, note)
        chunks = index_note(note)
    return {
        "updated": True,
        **_note_payload(note, res),
        "changed": changed,
        "chunks_indexed": chunks,
    }


@mcp.tool(annotations=_FORGET)
def forget_note(note_id: str, reason: str, tombstone: bool = True) -> dict:
    """Remove a note from memory: soft delete, never an unlink.

    The file moves to the vault's `.trash/` folder (undo: move it back and run
    `sift reindex`), its index rows are deleted, and - unless `tombstone=False` - its
    id (and, for an ingested note, its url) is recorded so a bulk re-ingest does not
    bring it back. Every file carrying the id is moved.

    Args:
        note_id: the exact note id (a slug is refused: deletes never guess).
        reason: why - stamped into the trashed copy.
        tombstone: keep ingests from re-adding it (default true).
    """
    note_id = (note_id or "").strip()
    reason = (reason or "").strip()
    if not note_id:
        raise ToolError("note_id is required")
    if not reason:
        raise ToolError("reason is required")

    from sift.quality import is_user_authored
    from sift.vault.notes import delete_note, load_note

    vault = _vault()
    errors: list[str] = []
    with _vault_lock(vault):
        cat = _catalog(vault, max_age=0.0)
        rows = cat.by_id(note_id)
        if not rows:
            near = cat.lookup(note_id)
            if near:
                ids = ", ".join(r.id for r in near[:10])
                raise ToolError(f"forget_note needs the exact note_id; {note_id!r} matches: {ids}")
            raise ToolError(f"no note with id {note_id!r}")
        urls: set[str] = set()
        for row in rows:
            try:
                meta = load_note(row.path).meta
            except Exception:  # noqa: BLE001 - unreadable: no url to tombstone
                continue
            # An ingested article's url blocks it from every feed; a user note's url
            # is only provenance, and the article itself stays ingestible.
            if meta.url and not is_user_authored(meta):
                urls.add(meta.url)
        moved = delete_note(note_id, vault=vault, reason=reason)
        if not moved:
            raise ToolError(f"no file carries {note_id!r} any more; nothing was moved")

        rows_deleted: int | None = None
        try:
            from sift.index.store import Store

            rows_deleted = Store().delete_notes([note_id])
        except Exception as exc:  # noqa: BLE001 - the move happened; report, don't raise
            errors.append(
                f"index: {_one_line(exc)} (its rows are reaped by the next `sift reindex`)"
            )
        tombstoned = False
        if tombstone:
            try:
                from sift.tombstones import record_tombstones

                record_tombstones(ids=[note_id], urls=sorted(urls), reason=f"forget_note: {reason}")
                tombstoned = True
            except Exception as exc:  # noqa: BLE001
                errors.append(f"tombstone: {_one_line(exc)}")
    out: dict[str, Any] = {
        "forgotten": True,
        "note_id": note_id,
        "trashed": [str(p) for p in moved],
        "index_rows_deleted": rows_deleted,
        "tombstoned": tombstoned,
        "hint": "Undo: move the file back out of .trash and run `sift reindex`.",
    }
    if errors:
        out["errors"] = errors
    return out


@mcp.tool(annotations=_FETCH)
def capture_url(
    url: str,
    program: str | None = None,
    tags: list[str] | None = None,
    force: bool = False,
) -> dict:
    """Fetch one article (a fresh writeup, an advisory) and keep it verbatim.

    Already in the vault: the existing note is returned and nothing is fetched. The
    page goes through the same extraction guards as the batch ingest sources (no
    JS-rendered shells, binaries or Medium browser spoofing), and an article
    published before the model's training cutoff is refused - the model already
    knows it.

    Args:
        url: http(s) URL of the article.
        program: bug bounty program / vendor it concerns.
        tags: extra tags.
        force: capture even if it is pre-cutoff or was pruned/forgotten before.
    """
    url = (url or "").strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        parts = None
    if parts is None or parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        raise ToolError("url must be an absolute http(s) URL")
    try:
        why = _local_host(parts.hostname)
    except ValueError:  # a malformed port or bracketed host
        why = "a malformed host"
    if why:
        # A page read mid-session can try to steer the agent at internal services.
        raise ToolError(f"refusing to fetch {url!r}: {why}")

    vault = _vault()
    existing = _note_with_url(vault, url)
    if existing is not None:
        return {"saved": False, "existing": True, **_row_payload(existing)}
    if not force:
        from sift.tombstones import load_tombstones

        if load_tombstones().has_url(url):
            return {
                "saved": False,
                "reason": "this url was pruned or forgotten before (tombstoned); "
                "pass force=True to capture it anyway",
            }

    try:
        from sift.ingest.single_url import fetch_url_note, is_pre_cutoff
    except ImportError as exc:
        raise ToolError(f"capture_url is unavailable: {_one_line(exc)}") from exc
    try:
        # Network I/O, never under the vault lock. The fetcher re-checks the host
        # (resolved) on the first request and on every redirect hop.
        note = fetch_url_note(url)
    except Exception as exc:  # noqa: BLE001 - a rejected page is an answer, not a crash
        reason = getattr(exc, "reason", None)  # single_url.CaptureError: short, content-free
        return {"saved": False, "url": url, "reason": str(reason or _one_line(exc))}
    if note is None:
        return {"saved": False, "url": url, "reason": "nothing to capture"}

    published = note.meta.created
    cutoff = get_settings().model_cutoff
    pre_cutoff = is_pre_cutoff(note)  # the batch sources' rule (SIFT_MODEL_CUTOFF)
    if pre_cutoff and not force:
        return {
            "saved": False,
            "url": url,
            "reason": "pre-cutoff",
            "published": published.isoformat(),
            "cutoff": cutoff.isoformat(),
            "hint": "published before the model's training cutoff, so it teaches nothing "
            "new; pass force=True if it really is missing",
        }
    if program:
        note.meta.program = program
    if tags:
        note.meta.tags = _union(note.meta.tags, tags)
    extra = dict(note.meta.extra) if isinstance(note.meta.extra, dict) else {}
    extra.setdefault("captured_via", "mcp-capture-url")  # bypassed the batch novelty gate
    note.meta.extra = extra
    if not note.meta.url:
        note.meta.url = url

    from sift.pipeline import index_note
    from sift.vault.notes import locate_note

    with _vault_lock(vault):
        # Another call may have captured it while this one was fetching.
        existing = _note_with_url(vault, note.meta.url) or _note_with_url(vault, url)
        if existing is not None:
            return {"saved": False, "existing": True, **_row_payload(existing)}
        if locate_note(vault, note.meta.id, max_age=0.0) is not None:
            # The (title-derived) id is held by a note with another url: never write
            # over it - an upsert would treat a same-title article as the same note.
            # Key this capture by its url instead, as the batch sources do.
            from sift.ingest.base import url_note_id

            note.meta.id = url_note_id(note.meta.id, note.meta.url)
        res = _write(vault, note, stamp=True)
        chunks = index_note(note)
    return {
        "saved": True,
        "new": res.created,
        **_note_payload(note, res),
        "url": note.meta.url,
        "created": published.isoformat() if published else None,
        "pre_cutoff": pre_cutoff,
        "chunks_indexed": chunks,
    }


def _row_payload(row: CatalogRow) -> dict:
    return {
        "note_id": row.id,
        "slug": row.slug,
        "title": row.title,
        "type": row.type,
        "created": row.created,
        "path": str(row.path),
    }


# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------


def _configure_logging() -> None:
    """Send the ``sift`` loggers to stderr - never stdout, the wire - unless the
    caller (the CLI) already configured them."""
    lg = logging.getLogger("sift")
    if lg.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("sift %(levelname)s %(name)s: %(message)s"))
    lg.addHandler(handler)
    if lg.level == logging.NOTSET:
        lg.setLevel(logging.WARNING)
    lg.propagate = False


def main() -> None:
    """Serve over stdio: what ``sift mcp`` and ``python -m sift.mcp_server`` run."""
    _configure_logging()
    # A stray print() anywhere then flushes at once into the fd stdio diverted to
    # stderr, instead of sitting in the buffer and landing on the wire at exit. Never
    # reassign sys.stdout: the SDK finds the wire through it.
    with contextlib.suppress(AttributeError, ValueError, OSError):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    enable_background()
    # No banner: Claude Code logs its stderr lines as errors, and skipping it also
    # skips FastMCP's PyPI update check before serving.
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
