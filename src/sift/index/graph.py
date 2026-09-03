"""Link-graph expansion — the Obsidian-like part.

Given a set of high-scoring notes, pull in their 1-hop ``[[wikilink]]`` /
``links:`` neighbours so the model sees connected context (e.g. a report links
the technique it used and the CVE it resembles).
"""

from __future__ import annotations

from pathlib import Path

from sift.vault.notes import Note, iter_notes


def build_link_index(vault: Path) -> dict[str, Note]:
    return {n.slug: n for n in iter_notes(vault)}


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
                if nb in seen or nb not in link_index:
                    continue
                seen.add(nb)
                out.append(nb)
                nxt.append(nb)
                if len(out) >= limit:
                    return out
        frontier = nxt
    return out
