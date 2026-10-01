"""Orchestration: vault notes -> chunks -> embeddings -> LanceDB, plus the
high-level search entrypoint used by both the CLI and the MCP server.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sift.config import get_settings
from sift.index import embed as _embed
from sift.index.graph import build_link_index, expand
from sift.index.rerank import get_reranker
from sift.index.store import ChunkRow, Hit, Store
from sift.quality import score_note
from sift.vault.chunk import chunk_markdown
from sift.vault.notes import Note, iter_notes


def _created_ts(note: Note) -> float:
    d = note.meta.created
    if not d:
        return 0.0
    return datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()


def _passage(title: str, heading: str, text: str) -> str:
    """Text we embed for a chunk — prefixed with the note's identity."""
    return f"{title}\n{heading}\n{text}" if heading else f"{title}\n{text}"


def _rows_for_note(note: Note, vectors: list[list[float]] | None = None) -> list[ChunkRow]:
    chunks = chunk_markdown(note.body)
    if not chunks:
        return []
    m = note.meta
    if vectors is None:
        passages = [_passage(m.title, c.heading, c.text) for c in chunks]
        vectors = _embed.get_embedder().embed(passages, kind="passage")
    mtime = note.path.stat().st_mtime if note.path and note.path.exists() else 0.0
    quality = score_note(m, note.body)
    created_ts = _created_ts(note)
    rows: list[ChunkRow] = []
    for c, vec in zip(chunks, vectors, strict=False):
        rows.append(
            ChunkRow(
                note_id=m.id,
                slug=note.slug,
                type=m.type,
                title=m.title,
                heading=c.heading,
                text=c.text,
                chunk_index=c.index,
                vector=vec,
                source=m.source or "",
                url=m.url or "",
                cwe=m.cwe,
                tags=m.tags,
                severity=m.severity or "",
                program=m.program or "",
                path=str(note.path) if note.path else "",
                mtime=mtime,
                quality=quality,
                created_ts=created_ts,
            )
        )
    return rows


def index_note(note: Note, store: Store | None = None) -> int:
    store = store or Store()
    rows = _rows_for_note(note)
    store.upsert_note(rows)
    return len(rows)


def index_notes(notes: list[Note], store: Store | None = None, *, replace: bool = True) -> int:
    """Embed and index many notes with a single batched embedding call."""
    store = store or Store()
    notes = [n for n in notes if n.body and n.body.strip()]
    if not notes:
        return 0
    per_note = [chunk_markdown(n.body) for n in notes]
    passages = [
        _passage(n.meta.title, c.heading, c.text)
        for n, chunks in zip(notes, per_note, strict=True)
        for c in chunks
    ]
    if not passages:
        return 0
    vectors = _embed.get_embedder().embed(passages, kind="passage")

    i = 0
    total = 0
    all_rows: list[ChunkRow] = []
    for n, chunks in zip(notes, per_note, strict=True):
        take = vectors[i : i + len(chunks)]
        i += len(chunks)
        if replace:
            store.delete_note(n.meta.id)
        rows = _rows_for_note(n, take)
        all_rows.extend(rows)
        total += len(rows)
    store.add_chunks(all_rows)
    return total


@dataclass
class ReindexStats:
    notes: int = 0
    chunks: int = 0
    skipped: int = 0
    # Notes whose file mtime matched the index, so they were not re-embedded.
    unchanged: int = 0
    by_type: dict[str, int] = field(default_factory=dict)


def count_notes(vault: Path | None = None) -> int:
    """How many notes `iter_notes` will yield. Used to size a progress bar."""
    from sift.vault.schema import NOTE_TYPES

    vault = vault or get_settings().resolved_vault()
    total = 0
    for note_type in NOTE_TYPES:
        d = vault / note_type
        if not d.is_dir():
            continue
        total += sum(
            1 for p in d.glob("*.md") if not p.name.startswith("_") and p.name != "README.md"
        )
    return total


def reindex(
    vault: Path | None = None,
    *,
    force: bool = False,
    batch: int = 512,
    on_progress: Callable[[int], None] | None = None,
) -> ReindexStats:
    """Rebuild the index from the vault, embedding chunks in large batches.

    `on_progress` receives the number of notes processed so far, after each flush.
    Kept as a callback so this module stays free of any UI dependency.
    """
    s = get_settings()
    vault = vault or s.resolved_vault()
    store = Store()
    if force:
        store.drop()

    stats = ReindexStats()
    embedder = _embed.get_embedder()

    # Without this, adding one note re-embedded all 9,000+ — minutes of GPU work to
    # index a single file. Chunk rows already carry the source file's mtime, so an
    # incremental pass can skip everything untouched. `--force` still rebuilds all.
    indexed_mtimes: dict[str, float] = {} if force else store.note_mtimes()

    # (note, [chunks]) buffered until we have `batch` chunks, then embed+write together.
    buf: list[tuple[Note, list]] = []
    buf_chunks = 0
    last_report = 0

    def flush() -> None:
        nonlocal buf, buf_chunks
        if not buf:
            return
        passages = [_passage(n.meta.title, c.heading, c.text) for n, chunks in buf for c in chunks]
        vectors = embedder.embed(passages, kind="passage")
        i = 0
        rows: list[ChunkRow] = []
        for n, chunks in buf:
            take = vectors[i : i + len(chunks)]
            i += len(chunks)
            if not force:
                store.delete_note(n.meta.id)
            rows.extend(_rows_for_note(n, take))
        store.add_chunks(rows)
        buf, buf_chunks = [], 0

    for note in iter_notes(vault):
        if indexed_mtimes and note.path is not None:
            try:
                on_disk = note.path.stat().st_mtime
            except OSError:
                on_disk = 0.0
            was = indexed_mtimes.get(note.meta.id)
            # Float equality is right here: both sides are the same stat value
            # round-tripped through the store, not a computed quantity.
            if was is not None and on_disk and was == on_disk:
                stats.unchanged += 1
                # Skipped notes still count as progress, or the bar sits at 0% while
                # racing through 9,000 unchanged files.
                if on_progress and stats.unchanged % 250 == 0:
                    on_progress(stats.notes + stats.unchanged)
                continue

        chunks = chunk_markdown(note.body)
        if not chunks:
            stats.skipped += 1
            continue
        buf.append((note, chunks))
        buf_chunks += len(chunks)
        stats.notes += 1
        stats.chunks += len(chunks)
        stats.by_type[note.meta.type] = stats.by_type.get(note.meta.type, 0) + 1
        if buf_chunks >= batch:
            flush()
            if on_progress:
                on_progress(stats.notes + stats.unchanged)
            elif stats.notes - last_report >= 2000:
                last_report = stats.notes
                print(f"  .. {stats.notes} notes indexed")

    flush()
    if on_progress:
        on_progress(stats.notes)
    store.ensure_fts()
    return stats


@dataclass
class SearchResult:
    hits: list[Hit]
    linked: list[dict]  # expanded neighbour notes: {slug, title, type, path, why}


def search(
    query: str,
    *,
    k: int = 8,
    filters: dict | None = None,
    expand_links: bool = False,
    min_quality: int = 0,
) -> SearchResult:
    store = Store()
    pool = max(40, k * 5)
    reranker = get_reranker()
    hits = store.search(
        query,
        k=max(k, 20) if reranker else k,
        filters=filters,
        pool=pool,
        min_quality=min_quality,
    )
    hits = reranker.rerank(query, hits, top_k=k) if reranker else hits[:k]

    linked: list[dict] = []
    if expand_links and hits:
        s = get_settings()
        link_index = build_link_index(s.resolved_vault())
        seed_slugs = [h.slug for h in hits]
        for nb_slug in expand(seed_slugs, link_index, hops=1, limit=12):
            nb = link_index.get(nb_slug)
            if not nb:
                continue
            linked.append(
                {
                    "slug": nb_slug,
                    "title": nb.meta.title,
                    "type": nb.meta.type,
                    "path": str(nb.path) if nb.path else "",
                    "url": nb.meta.url or "",
                }
            )
    return SearchResult(hits=hits, linked=linked)


def get_note_by_slug(slug: str) -> Note | None:
    from slugify import slugify

    from sift.vault.notes import load_note
    from sift.vault.schema import NOTE_TYPES

    vault = get_settings().resolved_vault()
    want = slugify(slug, max_length=80)

    # Fast path: canonical location vault/<type>/<slug>.md
    for t in NOTE_TYPES:
        p = vault / t / f"{want}.md"
        if p.exists():
            return load_note(p)
    # Any filename match anywhere in the vault
    for p in vault.rglob(f"{want}.md"):
        return load_note(p)
    # Slow path: match on id / computed slug
    for note in iter_notes(vault):
        if note.slug == want or note.meta.id == slug:
            return note
    return None
