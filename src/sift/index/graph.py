"""Link-graph expansion — the Obsidian-like part.

Given a set of high-scoring notes, pull in their 1-hop ``[[wikilink]]`` /
``links:`` neighbours so the model sees connected context (e.g. a report links
the technique it used and the CVE it resembles).
"""

from __future__ import annotations

import os
from pathlib import Path

from sift.vault.notes import Note, iter_notes
from sift.vault.schema import NOTE_TYPES

# Building this means parsing every note in the vault - measured at 3.4s over 9,020
# notes, which made `search_memory(expand_links=True)` 40x slower than a plain search.
# Cache it against a cheap fingerprint instead: a scandir stat walk costs milliseconds
# and still notices new, changed or deleted notes, so a write-back is picked up on the
# next query rather than serving a stale graph.
_CACHE: dict[str, tuple[tuple[int, float], dict[str, Note]]] = {}


def _fingerprint(vault: Path) -> tuple[int, float]:
    """(note count, newest mtime) — stat only, no parsing."""
    count = 0
    newest = 0.0
    for note_type in NOTE_TYPES:
        d = vault / note_type
        if not d.is_dir():
            continue
        try:
            entries = os.scandir(d)
        except OSError:  # pragma: no cover - vault removed mid-run
            continue
        with entries:
            for e in entries:
                if not e.name.endswith(".md") or e.name.startswith("_"):
                    continue
                count += 1
                try:
                    newest = max(newest, e.stat().st_mtime)
                except OSError:
                    continue
    return count, newest


# Note slugs carry a source prefix from their id ("tech-", "research-", "top10-"),
# but a `[[wikilink]]` is written from the human-readable title and almost never
# includes it. Every cross-link in the first batch of technique notes was broken this
# way - 0 of 5 resolved - so the link graph returned nothing. Register each note under
# its prefix-stripped slug too, without ever shadowing a real one.
_ID_PREFIXES = ("tech", "research", "top10", "writeup", "idea", "repo", "cve", "targ", "find", "writ")


def _aliases(slug: str) -> list[str]:
    for p in _ID_PREFIXES:
        if slug.startswith(p + "-"):
            return [slug.removeprefix(p + "-")]
    return []


def _build(vault: Path) -> dict[str, Note]:
    from slugify import slugify

    notes = list(iter_notes(vault))
    index = {n.slug: n for n in notes}
    for note in notes:
        # Filenames are now the note's title, so an Obsidian-style `[[Cookie sandwich]]`
        # link names the title rather than the slug. Register both, plus the
        # prefix-stripped form, without ever shadowing a real slug.
        aliases = [*_aliases(note.slug), slugify(note.meta.title, max_length=80)]
        if note.path:
            aliases.append(slugify(note.path.stem, max_length=80))
        for alias in aliases:
            if alias:
                index.setdefault(alias, note)
    return index


def build_link_index(vault: Path, *, use_cache: bool = True) -> dict[str, Note]:
    if not use_cache:
        return _build(vault)

    key = str(vault)
    fp = _fingerprint(vault)
    cached = _CACHE.get(key)
    if cached and cached[0] == fp:
        return cached[1]

    index = _build(vault)
    _CACHE[key] = (fp, index)
    return index


def clear_link_index_cache() -> None:
    """Drop the cache. Tests use this; production relies on the fingerprint."""
    _CACHE.clear()


def expand(
    seeds: list[str],
    link_index: dict[str, Note],
    *,
    hops: int = 1,
    limit: int = 12,
) -> list[str]:
    """Return neighbour slugs (excluding seeds), nearest first, capped at ``limit``."""
    seen = set(seeds)
    frontier = list(seeds)
    out: list[str] = []
    for _ in range(hops):
        nxt: list[str] = []
        for slug in frontier:
            note = link_index.get(slug)
            if not note:
                continue
            for nb in note.all_links():
                target = link_index.get(nb)
                if target is None:
                    continue
                # A wikilink may name an alias ("cookie-sandwich") rather than the
                # note's real slug ("tech-cookie-sandwich"). Emit the canonical slug
                # so callers can feed it straight back into get_note.
                canonical = target.slug
                if canonical in seen:
                    continue
                seen.add(canonical)
                seen.add(nb)
                out.append(canonical)
                nxt.append(canonical)
                if len(out) >= limit:
                    return out
        frontier = nxt
    return out
